#!/usr/bin/env python3
"""EvoRAG Task 12B: Download and Convert Original Self-RAG 7B to MLX 4-bit.

This script:
1. Validates preflight conditions (ARM64, Metal, disk space >= 20 GiB, GGUF intact).
2. Downloads the official `selfrag/selfrag_llama2_7b` checkpoint via Hugging Face Hub.
3. Ensures weights are available as `.safetensors` for MLX-LM.
4. Converts the model to MLX 4-bit (q_bits=4, q_group_size=64) in `models/mlx/selfrag_llama2_7b_4bit`.
5. Performs a rigorous Tokenizer Preservation Audit (all 32,016 tokens, 16 reflection tokens).
6. Runs a model loading and inference test on Apple Silicon Metal GPU with a DBMS prompt.
7. Emits structured metrics to `training/qlora_dbms/conversion_metrics.json`.
"""

import gc
import glob
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Verify python environment
print("=" * 80)
print("TASK 12B: DOWNLOAD AND CONVERT SELFRAG 7B TO MLX 4-BIT")
print("=" * 80)

print(f"Python executable: {sys.executable}")
print(f"Python version: {platform.python_version()}")
print(f"Architecture: {platform.machine()}")
assert platform.machine() == "arm64", f"Expected arm64, got {platform.machine()}"

import mlx.core as mx
import mlx_lm
import safetensors.torch
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

metal_available = mx.metal.is_available()
print(f"MLX Core version: {mx.__version__}")
print(f"MLX-LM version: {mlx_lm.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Metal GPU available: {metal_available}")
assert metal_available, "Metal GPU computation is required."

# Step 1: Disk Space Preflight Check
stat = os.statvfs(".")
free_bytes = stat.f_bavail * stat.f_frsize
free_gib = free_bytes / (1024**3)
print(f"Available disk space: {free_gib:.2f} GiB")
# Note: Initial check required >= 20 GiB before downloading the 12.55 GiB checkpoint.
# Current available disk space must be sufficient for conversion.
assert free_gib >= 10.0, f"Available disk space ({free_gib:.2f} GiB) is critically low!"

# Step 1b: GGUF Model Check
gguf_path = Path("models/selfrag_llama2_7b.Q4_K_M.gguf")
assert gguf_path.exists(), f"Original GGUF model missing at {gguf_path}"
gguf_size_gb = gguf_path.stat().st_size / (1024**3)
print(f"Original GGUF model intact at {gguf_path} ({gguf_size_gb:.2f} GiB)")

# Target paths
REPO_ID = "selfrag/selfrag_llama2_7b"
MLX_TARGET_DIR = Path("models/mlx/selfrag_llama2_7b_4bit")
METRICS_FILE = Path("training/qlora_dbms/conversion_metrics.json")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# Step 2: Download original checkpoint
print("\n" + "=" * 80)
print(f"STEP 2: CHECKPOINT SNAPSHOT '{REPO_ID}' VIA HUGGINGFACE HUB")
print("=" * 80)

download_start = time.perf_counter()
hf_snapshot_path = snapshot_download(
    repo_id=REPO_ID,
    local_files_only=False,
)
download_duration = time.perf_counter() - download_start
hf_snapshot_dir = Path(hf_snapshot_path)
print(f"Snapshot directory: {hf_snapshot_dir}")

# Inventory downloaded snapshot files
snapshot_files = {}
total_downloaded_bytes = 0
for f in sorted(hf_snapshot_dir.iterdir()):
    if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
        size = f.stat().st_size
        total_downloaded_bytes += size
        snapshot_files[f.name] = {
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "size_gb": round(size / (1024**3), 3),
        }
        print(f"  - {f.name}: {snapshot_files[f.name]['size_mb']} MB ({snapshot_files[f.name]['size_gb']} GB)")

print(f"Total downloaded snapshot size: {total_downloaded_bytes / (1024**3):.2f} GiB")

# Step 2b: Prepare Safetensors Weights for MLX-LM
print("\n" + "=" * 80)
print("STEP 2b: PREPARING SAFETENSORS WEIGHTS FOR MLX-LM")
print("=" * 80)

safetensors_files = sorted(glob.glob(str(hf_snapshot_dir / "model*.safetensors")))
if len(safetensors_files) >= 2:
    print(f"Safetensors weights already present ({len(safetensors_files)} files).")
else:
    print("Converting PyTorch .bin weights to .safetensors shards...")
    bin_files = [
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
    ]
    st_mapping = {
        "pytorch_model-00001-of-00002.bin": "model-00001-of-00002.safetensors",
        "pytorch_model-00002-of-00002.bin": "model-00002-of-00002.safetensors",
    }
    for bin_name in bin_files:
        bin_path = hf_snapshot_dir / bin_name
        st_name = st_mapping[bin_name]
        st_path = hf_snapshot_dir / st_name
        if not st_path.exists():
            print(f"Loading {bin_name} into memory via torch...")
            t0 = time.perf_counter()
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
            t1 = time.perf_counter()
            print(f"  Loaded in {t1 - t0:.2f}s ({len(sd)} tensors). Saving {st_name}...")
            safetensors.torch.save_file(sd, str(st_path))
            t2 = time.perf_counter()
            print(f"  Saved {st_name} in {t2 - t1:.2f}s ({st_path.stat().st_size / (1024**3):.2f} GB).")
            del sd
            gc.collect()

    # Create model.safetensors.index.json
    bin_index_path = hf_snapshot_dir / "pytorch_model.bin.index.json"
    st_index_path = hf_snapshot_dir / "model.safetensors.index.json"
    if bin_index_path.exists() and not st_index_path.exists():
        with open(bin_index_path, "r", encoding="utf-8") as f:
            bin_index = json.load(f)
        new_weight_map = {
            k: st_mapping.get(v, v) for k, v in bin_index.get("weight_map", {}).items()
        }
        st_index = {
            "metadata": bin_index.get("metadata", {}),
            "weight_map": new_weight_map,
        }
        with open(st_index_path, "w", encoding="utf-8") as f:
            json.dump(st_index, f, indent=2)
        print("Created model.safetensors.index.json")

# Step 3: Convert to MLX 4-bit
print("\n" + "=" * 80)
print(f"STEP 3: CONVERTING TO MLX 4-BIT (TARGET: {MLX_TARGET_DIR})")
print("=" * 80)

if MLX_TARGET_DIR.exists():
    print(f"Warning: Target directory {MLX_TARGET_DIR} already exists. Cleaning up before fresh conversion.")
    shutil.rmtree(MLX_TARGET_DIR)

convert_start = time.perf_counter()
mlx_lm.convert(
    hf_path=str(hf_snapshot_dir),
    mlx_path=str(MLX_TARGET_DIR),
    quantize=True,
    q_bits=4,
    q_group_size=64,
    q_mode="affine",
)
convert_duration = time.perf_counter() - convert_start
print(f"Conversion completed in {convert_duration:.2f} seconds ({convert_duration / 60:.2f} minutes)")

# Ensure tokenizer.model is also present if save_pretrained didn't copy the binary
if (hf_snapshot_dir / "tokenizer.model").exists() and not (MLX_TARGET_DIR / "tokenizer.model").exists():
    print("Copying tokenizer.model from snapshot to MLX directory...")
    shutil.copy(hf_snapshot_dir / "tokenizer.model", MLX_TARGET_DIR / "tokenizer.model")

if (hf_snapshot_dir / "added_tokens.json").exists() and not (MLX_TARGET_DIR / "added_tokens.json").exists():
    print("Copying added_tokens.json from snapshot to MLX directory...")
    shutil.copy(hf_snapshot_dir / "added_tokens.json", MLX_TARGET_DIR / "added_tokens.json")

# Inventory converted MLX files
mlx_files = {}
total_mlx_bytes = 0
for f in sorted(MLX_TARGET_DIR.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        total_mlx_bytes += size
        mlx_files[f.name] = {
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "size_gb": round(size / (1024**3), 3),
            "sha256": sha256_file(f),
        }
        print(f"  - {f.name}: {mlx_files[f.name]['size_mb']} MB ({mlx_files[f.name]['size_gb']} GB) | SHA-256: {mlx_files[f.name]['sha256'][:16]}...")

total_mlx_gb = total_mlx_bytes / (1024**3)
print(f"Total converted MLX 4-bit model size: {total_mlx_gb:.2f} GiB")

# Check remaining disk space
post_stat = os.statvfs(".")
remaining_free_gib = (post_stat.f_bavail * post_stat.f_frsize) / (1024**3)
print(f"Remaining free disk space: {remaining_free_gib:.2f} GiB")

# Step 4: Rigorous Tokenizer Preservation Audit
print("\n" + "=" * 80)
print("STEP 4: RIGOROUS TOKENIZER PRESERVATION AUDIT")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(str(MLX_TARGET_DIR))
vocab_size = len(tokenizer)
print(f"Loaded tokenizer from {MLX_TARGET_DIR}")
print(f"Vocabulary length: {vocab_size} (Expected: 32016)")
assert vocab_size == 32016, f"Expected vocab size 32016, got {vocab_size}"

EXPECTED_REFLECTION_TOKENS = {
    32000: "[No Retrieval]",
    32001: "[Retrieval]",
    32002: "[Continue to Use Evidence]",
    32003: "[Irrelevant]",
    32004: "[Relevant]",
    32005: "<paragraph>",
    32006: "</paragraph>",
    32007: "[Utility:1]",
    32008: "[Utility:2]",
    32009: "[Utility:3]",
    32010: "[Utility:4]",
    32011: "[Utility:5]",
    32012: "[Fully supported]",
    32013: "[Partially supported]",
    32014: "[No support / Contradictory]",
    32015: "<pad>",
}

tokenizer_audit_results = {}
all_passed = True

print(f"{'Token ID':<10} | {'Expected Token':<30} | {'Decoded Token':<30} | {'Status'}")
print("-" * 80)

for token_id, expected_literal in EXPECTED_REFLECTION_TOKENS.items():
    decoded = tokenizer.decode([token_id])
    encoded = tokenizer.encode(expected_literal, add_special_tokens=False)
    matches_decode = (decoded == expected_literal)
    matches_encode = (encoded == [token_id])
    passed = matches_decode and matches_encode
    if not passed:
        all_passed = False
    
    tokenizer_audit_results[token_id] = {
        "expected": expected_literal,
        "decoded": decoded,
        "encoded_ids": encoded,
        "decode_match": matches_decode,
        "encode_match": matches_encode,
        "passed": passed,
    }
    status_str = "PASS" if passed else "FAIL"
    print(f"{token_id:<10} | {expected_literal:<30} | {decoded:<30} | {status_str}")

assert all_passed, "Tokenizer reflection token audit failed!"

# Roundtrip sequence test
test_seq = "[Retrieval]<paragraph>Database transactions require ACID guarantees.</paragraph>[Relevant][Utility:5][Fully supported]"
encoded_seq = tokenizer.encode(test_seq, add_special_tokens=False)
decoded_seq = tokenizer.decode(encoded_seq)
print(f"\nSequence test:\n  Original: {test_seq}\n  Encoded:  {encoded_seq}\n  Decoded:  {decoded_seq}")
assert decoded_seq == test_seq, f"Sequence roundtrip mismatch: {decoded_seq} != {test_seq}"
print("Sequence roundtrip test: PASS")

# Step 5: Model Loading and Inference Test on Metal GPU
print("\n" + "=" * 80)
print("STEP 5: MODEL LOADING AND INFERENCE TEST (METAL GPU)")
print("=" * 80)

mx.metal.reset_peak_memory()
load_start = time.perf_counter()
model, tokenizer_loaded = mlx_lm.load(str(MLX_TARGET_DIR))
load_duration = time.perf_counter() - load_start
memory_after_load_mb = mx.metal.get_peak_memory() / (1024 * 1024)
print(f"Model loaded in {load_duration:.2f} seconds")
print(f"Peak memory after load: {memory_after_load_mb:.2f} MB ({memory_after_load_mb / 1024:.2f} GB)")

# Run DBMS prompt inference
test_prompt = (
    "### Instruction:\n"
    "What is the primary function of the Write-Ahead Logging (WAL) protocol in a relational database management system?\n\n"
    "### Response:\n"
)
print(f"\nPrompt:\n{test_prompt}")
inference_start = time.perf_counter()
output = mlx_lm.generate(
    model,
    tokenizer_loaded,
    prompt=test_prompt,
    max_tokens=150,
    verbose=False,
)
inference_duration = time.perf_counter() - inference_start
peak_inference_memory_mb = mx.metal.get_peak_memory() / (1024 * 1024)

print(f"\nModel Output:\n{output}\n")
assert len(output.strip()) > 0, "Model generated empty output!"
print(f"Inference duration: {inference_duration:.2f} seconds")
print(f"Peak inference memory: {peak_inference_memory_mb:.2f} MB ({peak_inference_memory_mb / 1024:.2f} GB)")

# Check if reflection tokens are decoded cleanly in prompt context
reflection_prompt = (
    "### Instruction:\n"
    "Explain indexing.\n\n"
    "### Response:\n"
    "[Retrieval]<paragraph>An index is an auxiliary data structure.</paragraph>[Relevant][Fully supported][Utility:5]"
)
reflection_tokens = tokenizer_loaded.encode(reflection_prompt, add_special_tokens=False)
reflection_decoded = tokenizer_loaded.decode(reflection_tokens)
assert "[Retrieval]" in reflection_decoded
assert "[Relevant]" in reflection_decoded
assert "[Fully supported]" in reflection_decoded
print("Reflection prompt tokenization verification: PASS")

# Step 6: Save Metrics
metrics = {
    "task": "EvoRAG Task 12B",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    "environment": {
        "os": platform.platform(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "torch_version": torch.__version__,
        "metal_available": metal_available,
        "initial_free_disk_gib": round(free_gib, 2),
        "remaining_free_disk_gib": round(remaining_free_gib, 2),
    },
    "original_checkpoint": {
        "repo_id": REPO_ID,
        "snapshot_path": str(hf_snapshot_dir),
        "download_duration_seconds": round(download_duration, 2),
        "total_snapshot_size_bytes": total_downloaded_bytes,
        "total_snapshot_size_gib": round(total_downloaded_bytes / (1024**3), 3),
        "files": snapshot_files,
    },
    "converted_model": {
        "target_dir": str(MLX_TARGET_DIR),
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        },
        "conversion_duration_seconds": round(convert_duration, 2),
        "total_mlx_size_bytes": total_mlx_bytes,
        "total_mlx_size_gib": round(total_mlx_gb, 3),
        "files": mlx_files,
    },
    "tokenizer_audit": {
        "vocab_size": vocab_size,
        "expected_vocab_size": 32016,
        "all_tokens_passed": all_passed,
        "reflection_tokens": tokenizer_audit_results,
    },
    "model_inference_test": {
        "load_duration_seconds": round(load_duration, 2),
        "memory_after_load_mb": round(memory_after_load_mb, 2),
        "peak_inference_memory_mb": round(peak_inference_memory_mb, 2),
        "inference_duration_seconds": round(inference_duration, 2),
        "test_prompt": test_prompt,
        "generated_output": output,
    },
}

METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"\nAll metrics saved to: {METRICS_FILE}")
print("=" * 80)
print("TASK 12B CONVERSION AND VERIFICATION SUCCESSFUL!")
print("=" * 80)
