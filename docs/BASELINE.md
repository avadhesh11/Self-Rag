# Self-RAG Reproducible Baseline Guide

This document establishes the exact configuration, dependencies, execution commands, and hardware constraints for running the Self-RAG baseline in this repository.

---

## 1. Pipeline Architecture & Components

* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (default, 384-dim, fast CPU execution).
* **Vector Store**: FAISS (`IndexFlatIP` with normalized vectors for cosine similarity).
* **Passage Retriever**: [`PassageRetriever`](../retriever/retriever.py) loads `meta.json`, `corpus.jsonl`, and `faiss.index`.
* **Generator Backend**: [`SelfRAGGenerator`](../selfrag/generator.py) supports two interchangeable backends:
  * `llamacpp` (Default): Local quantized GGUF models via `llama-cpp-python`.
  * `vllm`: Direct Hugging Face weights (`selfrag/selfrag_llama2_7b`) for Linux + NVIDIA GPUs with 14GB+ VRAM.
* **Inference Loop**: [`self_rag_answer`](../selfrag/pipeline.py) implements the Algorithm 1 multi-segment loop:
  1. Greedy decoding probe predicts initial retrieval decision (`[Retrieval]` vs `[No Retrieval]`).
  2. If `[Retrieval]`, retrieves $k$ passages and generates candidate responses in parallel/sequence.
  3. Critiques each candidate across `ISREL` (relevance), `ISSUP` (support / factuality), and `ISUSE` (utility).
  4. Selects the highest-scoring candidate and evaluates trailing tokens for continuation (`[Continue to Use Evidence]`, next `[Retrieval]`, or completion).

---

## 2. Dependencies & Environment

### Python Version
* Python 3.10 to 3.12 (Tested on Python 3.12.6, Windows x64).

### Core Retrieval & Indexing Requirements
```bash
pip install pypdf sentence-transformers faiss-cpu numpy python-dotenv
```

### Self-RAG Local Generator (GGUF via llama.cpp)
```bash
pip install llama-cpp-python huggingface-hub
```
*(On Windows with NVIDIA CUDA, pre-built CUDA wheels for `llama-cpp-python` can be installed to enable optional GPU acceleration).*

### Normal RAG (Gemini, for comparison mode)
```bash
pip install google-genai
```

---

## 3. Required Model Files and Environment Variables

### Model File
* **Filename**: `models/selfrag_llama2_7b.Q4_K_M.gguf` (~4.1 GB).
* **Download Command**:
  ```bash
  huggingface-cli download QuantFactory/selfrag_llama2_7b-GGUF selfrag_llama2_7b.Q4_K_M.gguf --local-dir models/
  ```

### Environment Variables
* `SELFRAG_BACKEND`: `"llamacpp"` (default) or `"vllm"`.
* `SELF_RAG_MODEL_PATH`: Path to the `.gguf` file (default: `models/selfrag_llama2_7b.Q4_K_M.gguf`).
* `INDEX_DIR`: Directory containing `faiss.index`, `corpus.jsonl`, and `meta.json` (default: `index/`).
* `GEMINI_API_KEY`: Required only for Normal RAG or comparison (`--mode both`) runs.

---

## 4. Execution Commands

### Step 1: Build the FAISS Index
Extracts text from source PDF(s), cleans OCR/ligatures, splits sentences (preserving legal/numerical references), merges into passage-sized chunks with overlap, tags deterministic chunk hashes (`chunk_hash`) and source file version (`source_version`), and builds a FAISS index:

```bash
python index/build_index.py --pdf data/sample.pdf --out index/
```

For multi-document indexing:
```bash
python index/build_index.py --pdf data/doc1.pdf data/doc2.pdf --out index/ --target-words 120 --overlap-words 30
```

**Artifacts Generated**:
* `index/faiss.index`: FAISS vector index of passage embeddings.
* `index/corpus.jsonl`: JSON Lines corpus with `text`, `page`, `source`, `source_version`, `chunk_hash`, and `chunk_id`.
* `index/meta.json`: Metadata with `embed_model`, `num_chunks`, `target_words`, `overlap_words`, `sources`, and `source_versions`.

### Step 2: Run Self-RAG Demo
Interactive query:
```bash
python demo.py --mode self "What are the four types of reflection tokens used in Self-RAG?"
```

Query skipping retrieval:
```bash
python demo.py --mode self "What is 2 + 2?"
```

Side-by-side comparison with Normal RAG:
```bash
python demo.py --mode both "What are the reflection tokens in Self-RAG?"
```

### Step 3: Run Evaluation Suite
Run benchmark questions through the pipeline and print a metrics table:
```bash
python evaluation/evaluate.py --mode self --index index/
```

---

## 5. Expected Outputs & Metrics

### Return Object (`SelfRAGResult`)
* `query`: Original query string.
* `retrieved`: Boolean indicating whether retrieval was executed.
* `decision`: Initial retrieval reflection token (`[Retrieval]`, `[No Retrieval]`, etc.).
* `candidates`: List of scored candidates from segment 1, each with:
  * `passage`: Dict containing passage `text`, `page`, `source`, `chunk_id`, `chunk_hash`, `source_version`.
  * `isrel`, `issup`, `isuse`: Parsed reflection tokens.
  * `score`: Composite critique score.
* `selected`: Chosen best candidate.
* `segments`: Full multi-segment trace list (`index`, `decision`, `evidence`, `candidates`, `selected`, `answer_text`).
* `final_answer`: Cleaned, reflection-token-stripped final answer.

### Measured Metrics
* **Latency**: Wall-clock seconds measured via `time.perf_counter()` across pipeline execution in `comparison/runner.py`.
* **Retrieval Count**: `len(r.candidates)` or segment evidence count, capturing passages retrieved per query.

---

## 6. Local Hardware Feasibility (Windows, GTX 1650 4GB VRAM, 16GB RAM)

### Assessment:
1. **vLLM Backend**: **Not supported**. vLLM does not run natively on Windows, and loading `selfrag_llama2_7b` in fp16 requires $\ge$ 14 GB VRAM (exceeds the 4 GB VRAM limit).
2. **llama.cpp Full GPU Offload (`n_gpu_layers=-1`)**: **Will OOM**. Offloading all 32 layers of a 7B model plus KV context requires ~5 GB VRAM, causing an out-of-memory error on a 4 GB GPU.
3. **llama.cpp CPU Execution (`n_gpu_layers=0`)**: **Fully Feasible**. The 4.1 GB Q4_K_M GGUF model fits comfortably within the 16 GB of system RAM. Generation is CPU-bound with typical throughput of ~3–8 tokens/sec.
4. **llama.cpp Partial GPU Offload (`n_gpu_layers=10..14`)**: **Feasible with caution**. Offloading 10–14 layers consumes ~1.8–2.3 GB VRAM, leaving headroom for the Windows display server and KV cache.

