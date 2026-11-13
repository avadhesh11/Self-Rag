#!/usr/bin/env python3
"""EvoRAG Task 12D: QLoRA Smoke Test Execution and Verification.

This script:
1. Validates environment (Apple Silicon Metal GPU, ARM64, disk space).
2. Computes initial SHA-256 of base model weights and frozen predictions.
3. Sets up LoRA layers on the 4-bit base model using MLX-LM 0.32.2.
4. Captures initial adapter weights to verify parameter change.
5. Runs 10 real optimization iterations with prompt masking and validation.
6. Asserts finite loss at every iteration and successful backpropagation.
7. Verifies adapter parameters have changed (delta > 0).
8. Verifies base model weights remain bitwise identical.
9. Performs an inference test with the saved adapter on Metal GPU.
10. Emits metrics to `training/qlora_dbms/smoke_test_metrics.json`.
"""

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import types
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import mlx_lm
from mlx_lm import load
from mlx_lm.utils import save_config
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import CacheDataset, TrainingArgs, TrainingCallback, default_loss, evaluate, iterate_batches, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

BASE_MODEL_DIR = Path("models/mlx/selfrag_llama2_7b_4bit")
BASE_WEIGHT_FILE = BASE_MODEL_DIR / "model.safetensors"
ADAPTER_DIR = Path("training/qlora_dbms/adapters/smoke_test")
DATASET_DIR = Path("training/qlora_dbms/mlx_dataset")
METRICS_FILE = Path("training/qlora_dbms/smoke_test_metrics.json")
FROZEN_CSV = Path("evaluation/predictions/frozen/predictions_20261109_151957Z.csv")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_free_disk_gib() -> float:
    stat = os.statvfs(".")
    return (stat.f_bavail * stat.f_frsize) / (1024**3)

class MetricsLogger(TrainingCallback):
    def __init__(self):
        self.train_reports = []
        self.val_reports = []

    def on_train_loss_report(self, info):
        self.train_reports.append(info)
        print(f"  [Iter {info['iteration']:2d}/10] Train Loss: {info['train_loss']:.4f} | "
              f"Tokens/s: {info['tokens_per_second']:.1f} | Peak Mem: {info['peak_memory']:.2f} GB")

    def on_val_loss_report(self, info):
        self.val_reports.append(info)
        print(f"  >>> [Validation Iter {info['iteration']:2d}] Val Loss: {info['val_loss']:.4f} (took {info['val_time']:.2f}s)")

def main():
    print("=" * 80)
    print("EVORAG TASK 12D: QLORA TRAINING SMOKE TEST")
    print("=" * 80)

    # 1. Environment & Preflight Checks
    print(f"Python: {sys.executable} ({platform.python_version()})")
    print(f"Platform: {platform.platform()} ({platform.machine()})")
    assert platform.machine() == "arm64", f"Expected arm64, got {platform.machine()}"
    
    metal_ok = mx.metal.is_available()
    print(f"MLX Core: {mx.__version__} | MLX-LM: {mlx_lm.__version__} | Metal: {metal_ok}")
    assert metal_ok, "Metal GPU acceleration is required."

    initial_free_disk = get_free_disk_gib()
    print(f"Initial Free Disk Space: {initial_free_disk:.2f} GiB")
    assert initial_free_disk >= 5.0, f"Critically low disk space: {initial_free_disk:.2f} GiB"

    print("Computing baseline base model weight hash...")
    t0_hash = time.perf_counter()
    initial_base_hash = sha256_file(BASE_WEIGHT_FILE)
    print(f"Base model SHA-256: {initial_base_hash} (verified in {time.perf_counter()-t0_hash:.2f}s)")

    initial_frozen_hash = sha256_file(FROZEN_CSV)
    print(f"Frozen predictions SHA-256: {initial_frozen_hash}")

    # Prepare adapter output directory
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Base Model and Apply LoRA Configuration
    print("\n--- Loading 4-Bit Base Model ---")
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    model, tokenizer = load(str(BASE_MODEL_DIR))
    load_duration = time.perf_counter() - load_start
    mem_after_load_gb = mx.get_peak_memory() / 1e9
    print(f"Model loaded in {load_duration:.2f}s | Memory after load: {mem_after_load_gb:.2f} GB")

    print("\n--- Applying LoRA Configuration (Rank 8, Scale 16.0, 16 Layers) ---")
    model.freeze()
    lora_config = {"rank": 8, "scale": 16.0, "dropout": 0.0}
    linear_to_lora_layers(
        model,
        num_layers=16,
        config=lora_config,
        use_dora=False,
    )
    print_trainable_parameters(model)

    # Collect initial adapter parameters snapshot
    trainable_params_before = dict(tree_flatten(model.trainable_parameters()))
    print(f"Trainable adapter parameter tensors: {len(trainable_params_before)}")
    sample_param_name = list(trainable_params_before.keys())[0]
    initial_sample_weights = mx.array(trainable_params_before[sample_param_name])
    initial_param_norm = mx.linalg.norm(initial_sample_weights).item()
    print(f"Sample adapter tensor '{sample_param_name}': shape={initial_sample_weights.shape}, L2-norm={initial_param_norm:.6f}")

    # 3. Load Datasets
    print("\n--- Loading Dataset Splits ---")
    dataset_args = types.SimpleNamespace(
        data=str(DATASET_DIR),
        hf_dataset=False,
        train=True,
        test=False,
        mask_prompt=True,
        prompt_feature="prompt",
        completion_feature="completion",
    )
    train_set, valid_set, _ = load_dataset(dataset_args, tokenizer)
    print(f"Train set: {len(train_set)} examples | Validation set: {len(valid_set)} examples")

    # 4. Initialize Optimizer and Training Loop
    print("\n--- Initializing Training Loop (10 Iterations) ---")
    learning_rate = 1.0e-5
    optimizer = optim.Adam(learning_rate=learning_rate)

    adapter_file = ADAPTER_DIR / "adapters.safetensors"
    config_dict = {
        "model": str(BASE_MODEL_DIR),
        "fine_tune_type": "lora",
        "num_layers": 16,
        "batch_size": 1,
        "iters": 10,
        "learning_rate": learning_rate,
        "lora_parameters": lora_config,
        "mask_prompt": True,
        "max_seq_length": 1024,
    }
    save_config(config_dict, ADAPTER_DIR / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=1,
        iters=10,
        val_batches=5,
        steps_per_report=1,
        steps_per_eval=5,
        steps_per_save=10,
        adapter_file=adapter_file,
        max_seq_length=1024,
        grad_checkpoint=False,
        grad_accumulation_steps=1,
    )

    logger = MetricsLogger()

    print("\n--- Starting Smoke Test Training Steps ---")
    train_start = time.perf_counter()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
        training_callback=logger,
    )
    train_duration = time.perf_counter() - train_start
    peak_training_mem_gb = mx.get_peak_memory() / 1e9

    print(f"\nTraining completed in {train_duration:.2f} seconds.")
    print(f"Peak Metal Memory during training: {peak_training_mem_gb:.2f} GB")

    # 5. Verify Training Outcomes
    print("\n--- Verifying Training Outcomes ---")
    assert len(logger.train_reports) == 10, f"Expected 10 training reports, got {len(logger.train_reports)}"
    initial_loss = logger.train_reports[0]["train_loss"]
    final_loss = logger.train_reports[-1]["train_loss"]
    print(f"Initial Step Loss (Step 1):  {initial_loss:.4f}")
    print(f"Final Step Loss   (Step 10): {final_loss:.4f}")

    for r in logger.train_reports:
        loss_val = r["train_loss"]
        assert not (loss_val != loss_val), f"NaN detected at step {r['iteration']}"
        assert abs(loss_val) != float("inf"), f"Inf detected at step {r['iteration']}"
    print("  [PASS] All 10 iteration losses are finite real values.")

    assert len(logger.val_reports) >= 2, f"Expected at least 2 validation evaluations, got {len(logger.val_reports)}"
    for vr in logger.val_reports:
        print(f"  [PASS] Validation evaluated: Iteration {vr['iteration']} -> Val Loss = {vr['val_loss']:.4f}")

    # Verify adapter weights have changed
    trainable_params_after = dict(tree_flatten(model.trainable_parameters()))
    final_sample_weights = mx.array(trainable_params_after[sample_param_name])
    weight_diff = mx.linalg.norm(final_sample_weights - initial_sample_weights).item()
    print(f"Sample adapter parameter '{sample_param_name}' weight delta: {weight_diff:.8f}")
    assert weight_diff > 0.0, f"Adapter weights did not change after 10 training iterations! (delta={weight_diff})"
    print("  [PASS] Adapter weights successfully updated via backpropagation.")

    # Verify saved adapter files
    assert adapter_file.exists(), f"Adapter file missing at {adapter_file}"
    adapter_size_mb = adapter_file.stat().st_size / (1024 * 1024)
    adapter_hash = sha256_file(adapter_file)
    print(f"Saved adapter file: {adapter_file} ({adapter_size_mb:.2f} MB, SHA-256: {adapter_hash[:16]}...)")
    assert (ADAPTER_DIR / "adapter_config.json").exists(), "Missing adapter_config.json"
    print("  [PASS] Adapter checkpoint files verified.")

    # Verify base model weights immutability
    print("\nVerifying base model weights immutability...")
    post_base_hash = sha256_file(BASE_WEIGHT_FILE)
    assert post_base_hash == initial_base_hash, f"Base model weights modified! Before: {initial_base_hash}, After: {post_base_hash}"
    print(f"  [PASS] Base model weights 100% UNCHANGED ({post_base_hash[:16]}...)")

    post_frozen_hash = sha256_file(FROZEN_CSV)
    assert post_frozen_hash == initial_frozen_hash, "Frozen predictions modified!"
    print(f"  [PASS] Frozen evaluation baseline CSV 100% UNCHANGED.")

    # 6. Post-Training Inference Test with Adapter
    print("\n--- Testing Model Inference with Loaded Adapter on Metal ---")
    mx.reset_peak_memory()
    inf_model, inf_tok = load(str(BASE_MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    
    test_prompt = (
        "### Instruction:\n"
        "Explain the write-ahead logging (WAL) protocol in database management systems.\n\n"
        "### Response:\n"
    )
    inf_t0 = time.perf_counter()
    inf_output = mlx_lm.generate(
        inf_model,
        inf_tok,
        prompt=test_prompt,
        max_tokens=80,
        verbose=False,
    )
    inf_duration = time.perf_counter() - inf_t0
    print(f"Inference generated in {inf_duration:.2f}s:")
    print(f"Output:\n{inf_output}\n")
    assert len(inf_output.strip()) > 0, "Adapted model produced empty output!"
    print("  [PASS] Inference with LoRA adapter succeeded on Metal GPU.")

    final_free_disk = get_free_disk_gib()
    print(f"\nFinal Free Disk Space: {final_free_disk:.2f} GiB (delta: {final_free_disk - initial_free_disk:.2f} GiB)")

    # 7. Save Metrics
    metrics = {
        "task": "EvoRAG Task 12D Smoke Test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "arch": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__,
            "metal_available": metal_ok,
            "initial_free_disk_gib": round(initial_free_disk, 2),
            "final_free_disk_gib": round(final_free_disk, 2),
        },
        "base_model": {
            "model_path": str(BASE_MODEL_DIR),
            "sha256": initial_base_hash,
            "immutability_verified": True,
            "load_duration_seconds": round(load_duration, 2),
            "memory_after_load_gb": round(mem_after_load_gb, 2),
        },
        "lora_configuration": {
            "num_layers": 16,
            "rank": 8,
            "scale": 16.0,
            "dropout": 0.0,
            "batch_size": 1,
            "learning_rate": learning_rate,
            "max_seq_length": 1024,
            "mask_prompt": True,
            "trainable_parameter_tensors": len(trainable_params_before),
            "sample_parameter_name": sample_param_name,
            "weight_delta_l2_norm": round(weight_diff, 8),
        },
        "training_execution": {
            "iterations": 10,
            "duration_seconds": round(train_duration, 2),
            "peak_memory_gb": round(peak_training_mem_gb, 2),
            "initial_train_loss": round(initial_loss, 4),
            "final_train_loss": round(final_loss, 4),
            "train_reports": logger.train_reports,
            "val_reports": logger.val_reports,
        },
        "adapter_artifacts": {
            "adapter_directory": str(ADAPTER_DIR),
            "adapter_file": str(adapter_file),
            "adapter_size_mb": round(adapter_size_mb, 2),
            "adapter_sha256": adapter_hash,
        },
        "post_training_inference": {
            "prompt": test_prompt,
            "generated_output": inf_output,
            "duration_seconds": round(inf_duration, 2),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE}")
    print("=" * 80)
    print("TASK 12D SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
