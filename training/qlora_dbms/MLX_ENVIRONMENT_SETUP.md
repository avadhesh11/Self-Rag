# Isolated MLX QLoRA Environment Setup Guide (Task 12A)

**Target System**: Apple Silicon M3 Pro (36 GB Unified Memory)
**Operating System**: macOS 15.5 (Build 24F74)
**Target Environment**: `.venv-mlx` (Native ARM64 Python 3.12.16)
**Original Production Environment**: `.venv` (Preserved intact and unmodified)

---

## 1. Overview & Isolation Policy

To support fine-tuning with Apple's **MLX-LM** framework without compromising the existing working EvoRAG application, an isolated virtual environment (`.venv-mlx`) was created.

### Key Invariants:
1. **Zero Contamination**: The production environment `.venv` remains untouched with zero modified or upgraded packages.
2. **Native ARM64**: `.venv-mlx` is built with native `arm64` Python 3.12, completely avoiding Rosetta x86_64 translation.
3. **No Heavy Cloud / CUDA Dependencies**: PyTorch CUDA, triton, or NVIDIA packages are strictly excluded.
4. **Data & Artifact Safety**: Existing production GGUF models, SQLite experience databases, FAISS indexes, frozen benchmark predictions, and model caches are preserved.

---

## 2. Installation Commands

The environment was created using Homebrew's native ARM64 Python 3.12:

```bash
# 1. Create the isolated virtual environment
/opt/homebrew/bin/python3.12 -m venv .venv-mlx

# 2. Activate the environment
source .venv-mlx/bin/activate

# 3. Upgrade pip and install official MLX-LM training dependencies
pip install --upgrade pip
pip install mlx-lm
```

---

## 3. Activation & Deactivation

### To Activate `.venv-mlx`:
```bash
source .venv-mlx/bin/activate
```
Verify native ARM64 execution:
```bash
python -c "import platform; print('Architecture:', platform.machine())"
# Output must be: Architecture: arm64
```

### To Deactivate:
```bash
deactivate
```

---

## 4. Exact Package Versions in `.venv-mlx`

The following packages are installed in `.venv-mlx` (verified via `pip list`):

| Package | Version | Purpose in MLX QLoRA Pipeline |
| :--- | :--- | :--- |
| **`mlx`** | `0.32.4` | Core Apple MLX array framework for Apple Silicon |
| **`mlx-lm`** | `0.32.2` | High-level LLM training, conversion, and QLoRA utilities |
| **`mlx-metal`** | `0.32.4` | Metal GPU compute backend and residency set manager |
| **`transformers`** | `5.22.2` | Model configurations and AutoTokenizer support |
| **`tokenizers`** | `0.23.3` | Fast Rust tokenization backend |
| **`sentencepiece`** | `0.2.3` | LLaMA SentencePiece BPE tokenizer support |
| **`safetensors`** | `0.8.0` | Zero-copy tensor serialization |
| **`numpy`** | `2.5.4` | Array processing |
| **`huggingface-hub`** | `2.4.0` | Weight downloading and snapshot utilities |
| **`pydantic`** | `2.14.3` | Schema validation |
| **`pyyaml`** | `6.0.3` | Training configuration files |

---

## 5. Verification Commands

Run the automated diagnostic script to verify hardware, MLX, Metal GPU execution, and disk safety:

```bash
.venv-mlx/bin/python training/qlora_dbms/check_mlx_environment.py
```

### Expected Output:
```text
======================================================================
EvoRAG MLX Environment & Hardware Diagnostic
======================================================================
Chip Architecture:      Apple M3 Pro
Python Architecture:    arm64 (Native ARM64)
macOS Version:          15.5
Python Runtime:         3.12.16 (.../.venv-mlx/bin/python)
System Memory:          36.00 GiB Unified Memory
Disk Space:             38.15 GiB available / 926.35 GiB total (95.9% used)
Safety Threshold:       15.0 GiB minimum free space required
  -> Disk Safety Check:  [PASS] (38.15 GiB >= 15.0 GiB)
----------------------------------------------------------------------
Verifying MLX & MLX-LM Libraries...
MLX Core Version:       0.32.4
MLX-LM Version:         0.32.2
Default MLX Device:     Device(gpu, 0)
Metal GPU Available:    True
----------------------------------------------------------------------
Executing Minimal Metal Compute Operations...
  -> Tensor shape verification: a=(512, 512), b=(512, 512), matmul=(512, 512)
  -> nn.Linear forward pass:    input=(512, 512) -> output=(512, 128)
  -> Metal computation time:    ~128 ms
  -> Output tensor mean:        0.159470
======================================================================
MLX Environment Verification: SUCCESS
All core MLX, MLX-LM, and Metal GPU requirements satisfied.
======================================================================
```

### Configurable Disk Space Threshold:
To test or enforce a higher minimum threshold (e.g. 25 GiB):
```bash
.venv-mlx/bin/python training/qlora_dbms/check_mlx_environment.py --min-disk-gb 25.0
```

---

## 6. Troubleshooting & Sandbox Notes

1. **`RuntimeError: [Event::Event] Failed to create Metal shared event`**:
   - **Cause**: When running inside restricted IDE sandbox environments, macOS Mach/IOKit IPC boundaries restrict shared event creation between sandboxed processes and the Metal driver.
   - **Resolution**: Run commands natively in a standard terminal shell or with bypass sandbox permissions. Metal GPU execution succeeds natively without error outside the sandbox.
2. **Rosetta Architecture Conflict**:
   - **Symptom**: `platform.machine()` returns `x86_64` or `Illegal instruction`.
   - **Resolution**: Always launch python with native arm64 prefix: `arch -arm64 .venv-mlx/bin/python ...`.
3. **Disk Space Warning**:
   - If free disk space drops below 15 GiB, prune unused Hugging Face caches (`huggingface-cli delete-cache` or removing raw download snapshots) before initiating weight conversion.
4. **Verifying Production Isolation**:
   - Running `.venv/bin/python -c "import mlx"` will confirm `ModuleNotFoundError`, ensuring the production application environment remains cleanly separated.

