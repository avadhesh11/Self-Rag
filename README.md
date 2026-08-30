# Self-RAG: Explain by Implementation

A minimal, from-scratch reimplementation of the *control loop* from
**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
(Asai et al., ICLR 2024), wired up against the official `selfrag/selfrag_llama2_7b`
checkpoint and a small demo PDF.

## What Self-RAG actually is (per the paper)

Not just "retrieve then generate." The generator is fine-tuned to emit four
kinds of reflection tokens inline with its output:

| Token | What it decides |
|---|---|
| `[Retrieval]` / `[No Retrieval]` / `[Continue to Use Evidence]` | whether retrieval is needed at all, adaptively, per segment |
| `[Relevant]` / `[Irrelevant]` (ISREL) | is a retrieved passage actually relevant to the query |
| `[Fully supported]` / `[Partially supported]` / `[No support / Contradictory]` (ISSUP) | is the generated text actually backed by that passage |
| `[Utility:1..5]` (ISUSE) | how useful is the final answer overall |

The model generates one candidate continuation **per retrieved passage in
parallel**, scores each with these tokens, and the best-scoring candidate
wins. That selection loop is the actual contribution — the retriever and
base LM are comparatively replaceable.

## Project structure

```
self-rag-demo/
├── data/sample.pdf              # small demo document (Amazon rainforest facts)
├── index/
│   ├── build_index.py           # PDF -> sentence chunks -> passages -> FAISS
│   ├── corpus.jsonl             # generated
│   └── faiss.index              # generated
├── retriever/retriever.py       # embed query, top-k search over the index
├── selfrag/
│   ├── reflection_tokens.py     # exact token vocab + parsing + scoring
│   ├── generator.py             # vLLM wrapper, official prompt format
│   └── pipeline.py              # the actual Self-RAG control loop
├── demo.py                      # CLI: question -> full trace -> answer
├── requirements.txt             # retriever side (works on any machine)
├── requirements-mac.txt         # + llama.cpp/Metal generator (Apple silicon)
└── requirements-gpu.txt         # + vLLM generator (Linux + NVIDIA only)
```

Retriever, generator, and control loop are three separate, swappable
pieces on purpose — that's what makes this easy to extend later (different
embedding model, different vector store, multi-segment long-form
generation, a different base checkpoint, a UI, etc.) without touching the
others.

## What's been verified so far, and how

This was built and tested in stages, since the full pipeline needs a GPU
that this authoring environment doesn't have:

1. **PDF chunking (`index/build_index.py`)** — tested directly against
   `data/sample.pdf`. Caught and fixed a real bug: pypdf's text extraction
   does not reliably preserve blank-line paragraph breaks, so splitting on
   blank lines silently collapsed the whole page into one chunk. Fixed by
   splitting at the sentence level and merging up to a target word count
   instead — more robust for arbitrary real-world PDFs.
2. **FAISS indexing/retrieval wiring** — tested end-to-end with a
   placeholder (hash-based) embedder, since this sandbox can't reach
   huggingface.co to download the real sentence-transformers model. The
   indexing and search logic itself is confirmed correct; on your machine,
   `build_index.py` will download the real embedding model and just work.
3. **The Self-RAG control loop (`selfrag/pipeline.py`)** — unit-tested
   against stub generator/retriever objects that mimic the checkpoint's
   output format. Confirmed:
   - the no-retrieval branch short-circuits correctly for a query that
     doesn't need lookup,
   - an irrelevant retrieved passage (`[Irrelevant]`) is correctly
     excluded from candidate selection,
   - among relevant candidates, the one with better ISSUP/ISUSE scores
     is correctly selected over a merely "partially supported" one.
4. **`selfrag/generator.py`** — prompt format and sampling settings copied
   verbatim from the official model card's usage example
   (`skip_special_tokens=False`, `temperature=0.0`). Two backends behind
   one interface: llama.cpp/Metal (Apple silicon, default) and vLLM
   (Linux + NVIDIA).
5. **Generator + full pipeline live on Apple silicon** — run on an M3 Pro
   with the official checkpoint as a Q4_K_M GGUF
   (`QuantFactory/selfrag_llama2_7b-GGUF`). All three retrieval-decision
   modes the checkpoint actually exhibits were observed in its raw
   output: `[No Retrieval]` for a poem request, `[Retrieval]` for a
   factoid query, and no token at all for arithmetic/trivia — it answers
   directly and scores itself `[Utility:5]`. The control loop respects
   all three (see `_decide_retrieve`). Reflection tokens survive
   llama.cpp tokenization — they are plain added tokens in the vocab,
   not flagged special. Caveat: Q4 quantization can shift near-tie
   logits, so exact decision/score behavior may differ slightly from
   fp16; Q8_0 is in the same repo for higher fidelity.

## Running it for real

The official checkpoint is a plain `LlamaForCausalLM`, so it runs two ways:

- **Apple silicon / CPU**: llama.cpp over Metal from a GGUF conversion —
  the default backend (`SELFRAG_BACKEND=llamacpp`).
- **Linux + NVIDIA GPU**: vLLM loads the HF checkpoint directly
  (`SELFRAG_BACKEND=vllm`, ~14GB+ VRAM for fp16 7B).

### On a Mac (Apple silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt

# Download a GGUF conversion of the official checkpoint (~3.9 GB).
# Q4_K_M is the sweet spot for an 18 GB Mac; Q8_0 (6.8 GB) is in the
# same repo if you want maximum fidelity instead.
hf download QuantFactory/selfrag_llama2_7b-GGUF \
    selfrag_llama2_7b.Q4_K_M.gguf --local-dir models/

python index/build_index.py --pdf data/sample.pdf --out index/
python demo.py "How large is the Amazon rainforest?"
python demo.py "What is 2+2?"          # model answers directly, no retrieval
```

### On a Linux machine with an NVIDIA GPU

```bash
pip install -r requirements-gpu.txt

python index/build_index.py --pdf data/sample.pdf --out index/
SELFRAG_BACKEND=vllm python demo.py --model selfrag/selfrag_llama2_7b \
    "How large is the Amazon rainforest?"
```

> **Why vLLM won't install on a Mac.** vLLM ships Linux+NVIDIA builds only
> — there is no macOS wheel, which is the metadata/version error pip raises
> when it tries to install it here. That is not a version-pinning problem;
> no pin fixes it. This repo defaults to llama.cpp for exactly that reason.

## Known simplifications (documented, not hidden)

- **Single-segment generation.** The paper's full algorithm is recursive
  over multiple output segments for long-form answers. This demo does one
  retrieval decision + one round of candidate scoring, which is enough to
  demonstrate every piece of the control loop on short-answer queries.
  Extending to multi-segment generation is a natural next feature.
- **Simplified critique-score weighting.** The paper's actual scoring is a
  weighted combination of segment-level log-probabilities across the full
  reflection-token sequence (see paper Section 3.3 / Appendix). This demo
  uses a simpler linear combination of ISREL/ISSUP/ISUSE for candidate
  ranking — good enough to show *which* passage wins and *why*, not a
  reproduction of paper benchmark numbers.
- **Substitutable embedding model.** The paper uses Contriever-MSMARCO for
  retrieval; this demo defaults to `all-MiniLM-L6-v2` via
  sentence-transformers for a lighter CPU-friendly demo. Swap the model
  name in `build_index.py`/`retriever.py` if you want to match the paper
  more closely.


## Acknowledgements

This project builds upon the original implementation of **SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**.

The original SELF-RAG framework and implementation were developed by:

- Akari Asai
- Zeqiu Wu
- Yizhong Wang
- Avirup Sil
- Hannaneh Hajishirzi

Original repository:
https://github.com/AkariAsai/self-rag

Original paper:
"SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
(ICLR 2024)

Our project uses, adapts and **extends** the original implementation for academic purposes.
All credit for the original SELF-RAG methodology and implementation belongs to the respective authors. Our contributions are limited to the modifications, integrations, experiments, and/or features developed specifically for this project.
