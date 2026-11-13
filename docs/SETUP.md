# EvoRAG Setup and Environment Guide

## 1. Environment Architecture

EvoRAG maintains two cleanly isolated virtual environments to ensure stability between production GGUF inference and experimental Apple Silicon MLX training/inference:

| Environment | Path | Python Version | Primary Libraries | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Production Environment** | `.venv` | Python 3.13 (ARM64) | `llama-cpp-python`, `torch`, `sentence-transformers`, `faiss-cpu`, `google-genai` | Runs Normal RAG, baseline Self-RAG (GGUF), and GGUF EvoRAG. |
| **Experimental MLX Environment** | `.venv-mlx` | Python 3.12 (ARM64) | `mlx` (0.32.4), `mlx-lm` (0.32.2), `sentence-transformers`, `faiss-cpu`, `torch` | Runs local QLoRA fine-tuning and fine-tuned MLX EvoRAG. |

> [!NOTE]
> The unified CLI (`demo.py`) includes a controlled environment launcher. When invoked from `.venv`, selecting `--mode evorag-mlx` automatically dispatches to `.venv-mlx` via `arch -arm64` subprocess, and vice-versa.

---

## 2. Setting Up the Production Environment (`.venv`)

```bash
# 1. Create and activate Python 3.13 environment
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. For Apple Silicon Metal acceleration in llama-cpp-python:
CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

---

## 3. Setting Up the MLX Environment (`.venv-mlx`)

```bash
# 1. Create and activate Python 3.12 environment (native ARM64)
arch -arm64 python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate

# 2. Install MLX and fine-tuning packages
pip install --upgrade pip
pip install mlx==0.32.4 mlx-lm==0.32.2
pip install sentence-transformers faiss-cpu pypdf pyyaml pytest
```

---

## 4. Models and Checkpoints

### 4.1 Production GGUF Model
- **File**: `models/selfrag_llama2_7b.Q4_K_M.gguf`
- **Size**: 4,081,095,776 bytes (3.80 GiB)
- **SHA-256**: `fb39ef3293bffcac2c44415123ec14d7b62c5cdc46a5dbfab9cf3830afe7dc03`

### 4.2 Converted MLX 4-bit Base Model
- **Directory**: `models/mlx/selfrag_llama2_7b_4bit/`
- **Weights File**: `models/mlx/selfrag_llama2_7b_4bit/model.safetensors`
- **Size**: 3,790,899,952 bytes (3.53 GiB)
- **SHA-256**: `71576febb874c07b71697976f0892c2bab9e972e77bc880dcc0210eb6342db32`
- **Conversion Command**:
  ```bash
  source .venv-mlx/bin/activate
  python training/qlora_dbms/convert_selfrag.py \
      --hf-repo selfrag/selfrag_llama2_7b \
      --out-dir models/mlx/selfrag_llama2_7b_4bit \
      --q-bits 4
  ```

### 4.3 Fine-Tuned MLX QLoRA Adapter
- **Directory**: `training/qlora_dbms/adapters/full_dbms_v1/`
- **Weights File**: `training/qlora_dbms/adapters/full_dbms_v1/adapters.safetensors`
- **Size**: 40,001,186 bytes (38.15 MB)
- **Best Iteration**: 545 (Validation Loss: 0.8713, PPL: 2.39)

---

## 5. Knowledge Corpus and Memory Indices

### 5.1 Document Index (DBMS 15-PDF Corpus)
The dedicated DBMS FAISS index is located at `index/dbms_sql_v1/`:
```bash
# To rebuild index from source PDFs if needed:
source .venv/bin/activate
python index/build_index.py --pdf-dir EvoRAG_DBMS_SQL_15_Document_Corpus/pdf --out-dir index/dbms_sql_v1
```

### 5.2 Experience Memory
- **SQLite Database**: `data/experience_memory.db` (Contains 25 approved DBMS experiences)
- **FAISS Memory Index**: `data/experience_faiss.index` + `data/experience_meta.json` (25 vectors, 384 dimensions)

