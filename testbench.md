# Testbench — recreating the paper's benchmark with local models

The original Self-RAG eval data (PopQA / TriviaQA / PubHealth / ARC-C with
pre-retrieved Wikipedia passages) isn't publicly downloadable, so this repo
now has a synthetic equivalent: **55 questions over 4 fictional PDFs + the
Self-RAG paper** (`benchmark_addon/`), scored the way Table 2 of
2310.11511v1 is scored. A `demo.py` option for Ollama/local models was added
alongside it, and everything was verified end-to-end on this machine
(2026-10-30).

## What was added / changed

| Piece | What it is |
|---|---|
| `benchmark_addon/testbench/` | 4 generated fictional PDFs + `questions.jsonl` (55 questions: short / multi / tf / mc / unanswerable / general / paper). Fictional entities guarantee a no-retrieval baseline scores ~0 and "unanswerable" really is unanswerable. |
| `benchmark_addon/benchmark/` | Benchmark driver: `run_benchmark.py` (resumable, prints a Table-2-style report), `systems.py` (uniform wrappers for Self-RAG and Ollama baselines), `ollama_generator.py` (paper-style greedy decoding), `tasks.py` (loaders + retrieval attach + hit-rate check), `metrics.py` (official substring-match metric + closed-set parsing). |
| `benchmark_addon/index/` | FAISS index for the testbench: 371 passages (21 testbench + 350 paper) at ~40 words, 0 overlap, so retrieval must actually discriminate. |
| `normal_rag/ollama_generator.py` | `OllamaGenerator` with the same `generate(query, chunks)` interface as the Gemini generator — drops into the existing normal-RAG pipeline unchanged. |
| `demo.py` / `comparison/runner.py` | New mode **`ollama`** (menu option 4): local model + the shared retriever. `--ollama-model` / `--ollama-host` flags, `OLLAMA_MODEL` / `OLLAMA_HOST` env vars; default `llama3:8b` @ `localhost:11434`. |
| `selfrag/pipeline.py` | Applied `selfrag_pipeline.patch`: `self_rag_answer(normalize=False)` for benchmarks (the query normalizer collapses ARC's multi-line options). |
| Fixes | `run_benchmark.py` puts the repo root on `sys.path` and defaults `--selfrag-model` to the repo-root `models/` path; `build_testbench.py` resolves outputs next to itself (not cwd); `reportlab` added to `requirements.txt`. |

## Results (full run, 2026-10-30)

7 systems × all 55 questions, `--k 5`, paper substring-match metric:

| System | Short | Multi | T/F | MC | Unans | General | Paper | Avg |
|---|---|---|---|---|---|---|---|---|
| llama3:8b+ret | 100.0 | 88.9 | 85.7 | 100.0 | 33.3 | 50.0 | 66.7 | **74.9** |
| gemma3:12b+ret | 100.0 | 77.8 | 85.7 | 100.0 | 50.0 | 25.0 | 66.7 | **72.2** |
| llama2:7b+ret | 100.0 | 44.4 | 57.1 | 100.0 | 33.3 | 25.0 | 66.7 | 60.9 |
| llama2:7b | 11.1 | 0.0 | 57.1 | 0.0 | 100.0 | 100.0 | 16.7 | 40.7 |
| selfrag-adaptive | 55.6 | 0.0 | 57.1 | 20.0 | 16.7 | 100.0 | 0.0 | 35.6 |
| llama3:8b | 5.6 | 0.0 | 71.4 | 40.0 | 0.0 | 100.0 | 16.7 | 33.4 |
| gemma3:12b | 5.6 | 0.0 | 85.7 | 20.0 | 0.0 | 100.0 | 0.0 | 30.2 |

Retrieval hit-rate (gold evidence phrase inside the top-5 passages):
**96% (43/45)** — 100% on short/multi/tf/mc; the 2 misses are paper-type
questions where the all-MiniLM-L6-v2 embedder loses to the 350-chunk paper
corpus (see `benchmark_addon/testbench/README.md`).

The story matches the paper's: retrieval turns ~0-11% on fictional facts
into 100% for every model; no-retrieval wins on general knowledge;
the unanswerable column catches with-retrieval hallucinations (e.g. a model
inventing "Aurora Cell Technologies was founded in 2023"). Known gaps,
spot-checked in the raw outputs:
- **Self-RAG multi (0/9)**: the 7B checkpoint retrieves both facts but won't
  do the arithmetic (e.g. "268 metres / 214 metres" but no "54"), plus an
  occasional number hallucination even with the evidence in context
  (kv02: "1964" → "1982") and occasional refusals (az06). Run it with
  `--selfrag-mode always` for a forced-retrieval variant.
- **Unanswerable** is the weakest column for +ret models (33-50%); the
  abstention scorer is phrase-based, so eyeball
  `benchmark_results_pdf/*__unanswerable.jsonl` for the real behavior.

Per-question raw outputs: `benchmark_addon/benchmark_results_pdf/<system>__<type>.jsonl`.

## Commands to recreate

Prereqs: `.venv` (requirements + `reportlab`), `ollama serve` running with
the models pulled, and the Self-RAG gguf in `models/`.

```bash
source .venv/bin/activate
ollama pull llama2:7b llama3:8b gemma3:12b     # once

# 1. (Re)generate the testbench PDFs + questions (works from any cwd)
python benchmark_addon/testbench/build_testbench.py

# 2. Build the small-chunk testbench index (testbench PDFs + the paper)
python index/build_index.py \
    --pdf benchmark_addon/data/testbench/*.pdf data/2310.11511v1.pdf \
    --out benchmark_addon/index --target-words 40 --overlap-words 0

# 3. Run the benchmark (resumable -- safe to Ctrl-C and re-run) and print the table
cd benchmark_addon
python -m benchmark.run_benchmark run \
    --systems selfrag ollama:llama2:7b ollama:llama3:8b ollama:gemma3:12b \
    --retrieval both --k 5

# 4. Reprint the table / re-run subsets anytime
python -m benchmark.run_benchmark report
python -m benchmark.run_benchmark run --systems ollama:llama3:8b --retrieval on --limit 10 --seed 1
```

Retrieval sanity check alone (no models involved):

```bash
cd benchmark_addon && python - <<'EOF'
import sys; sys.path.insert(0, ".")
from benchmark.tasks import load_pdf_testbench, attach_retrieval, retrieval_hit
by_type = load_pdf_testbench("testbench/questions.jsonl")
attach_retrieval(by_type, "index", 5, "/tmp/retrieval_check.json", refresh=True)
hits = [h for items in by_type.values() for it in items if (h := retrieval_hit(it)) is not None]
print(f"hit-rate {100*sum(hits)/len(hits):.0f}% ({sum(hits)}/{len(hits)})")
EOF
```

Interactive demo with the local model (menu option 4, or directly):

```bash
python demo.py                                    # pick 4. Ollama RAG
python demo.py --mode ollama "Who founded the town of Kestrel Valley?"
python demo.py --mode ollama --ollama-model gemma3:12b "How large is the Amazon rainforest?"
```

The Ollama demo mode answers against the **main** `index/` (sample.pdf +
paper + testbench PDFs); the benchmark uses its own `benchmark_addon/index`.
