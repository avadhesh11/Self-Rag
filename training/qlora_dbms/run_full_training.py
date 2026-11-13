#!/usr/bin/env python3
"""EvoRAG Task 12E: Controlled Full DBMS QLoRA Training.

This script executes the full QLoRA fine-tuning run on Apple Silicon Metal GPU:
1. Performs mandatory preflight checks (hashes, disk space, smoke-test preservation).
2. Applies LoRA (rank 8, scale 16, 16 layers) to the 4-bit Self-RAG base model.
3. Trains for 3 epochs (546 iterations, batch size 1, lr 1e-5, prompt masking).
4. Evaluates validation loss every half-epoch (steps_per_eval=91) over full validation set.
5. Automatically saves epoch checkpoints and isolates the best checkpoint.
6. Asserts loss finiteness, checks base model immutability, and logs detailed metrics.
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# File Paths
BASE_MODEL_DIR = PROJECT_ROOT / "models/mlx/selfrag_llama2_7b_4bit"
BASE_WEIGHT_FILE = BASE_MODEL_DIR / "model.safetensors"
EXPECTED_BASE_SHA256 = "71576febb874c07b71697976f0892c2bab9e972e77bc880dcc0210eb6342db32"

DATASET_DIR = PROJECT_ROOT / "training/qlora_dbms/mlx_dataset"
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"

SMOKE_ADAPTER_FILE = PROJECT_ROOT / "training/qlora_dbms/adapters/smoke_test/adapters.safetensors"
OUTPUT_ADAPTER_DIR = PROJECT_ROOT / "training/qlora_dbms/adapters/full_dbms_v1"
METRICS_FILE = PROJECT_ROOT / "training/qlora_dbms/full_training_metrics.json"

FROZEN_CSV = PROJECT_ROOT / "evaluation/predictions/frozen/predictions_20261109_151957Z.csv"
EXPECTED_FROZEN_SHA256 = "a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7"
GGUF_MODEL = PROJECT_ROOT / "models/selfrag_llama2_7b.Q4_K_M.gguf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_free_disk_gib() -> float:
    stat = os.statvfs(str(PROJECT_ROOT))
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


class FullTrainingCallback(TrainingCallback):
    def __init__(self, model, adapter_dir: Path, patience: int = 4):
        self.model = model
        self.adapter_dir = adapter_dir
        self.patience = patience
        self.train_reports = []
        self.val_reports = []
        self.best_val_loss = float("inf")
        self.best_iteration = -1
        self.worse_count = 0

    def on_train_loss_report(self, info):
        self.train_reports.append(info)
        loss = info["train_loss"]
        if loss != loss or abs(loss) == float("inf"):
            raise ValueError(f"Divergence detected: train_loss={loss} at iteration {info['iteration']}")
        print(
            f"  [Iter {info['iteration']:3d}/546] Train Loss: {loss:.4f} | "
            f"Tokens/s: {info['tokens_per_second']:5.1f} | Peak Mem: {info['peak_memory']:.2f} GB"
        )

    def on_val_loss_report(self, info):
        self.val_reports.append(info)
        val_loss = info["val_loss"]
        iteration = info["iteration"]
        print(
            f"  >>> [Validation Eval Iter {iteration:3d}] Val Loss: {val_loss:.4f} "
            f"(took {info['val_time']:.2f}s)"
        )

        if val_loss != val_loss or abs(val_loss) == float("inf"):
            raise ValueError(f"Divergence detected: val_loss={val_loss} at iteration {iteration}")

        if val_loss < self.best_val_loss:
            improvement = self.best_val_loss - val_loss
            self.best_val_loss = val_loss
            self.best_iteration = iteration
            self.worse_count = 0
            best_adapter_path = self.adapter_dir / "best_adapters.safetensors"
            current_weights = dict(tree_flatten(self.model.trainable_parameters()))
            mx.save_safetensors(str(best_adapter_path), current_weights)
            print(f"      * New best validation loss: {val_loss:.4f} (improved by {improvement:.4f}). Saved to {best_adapter_path.name}")
        else:
            self.worse_count += 1
            print(f"      - Val loss did not improve (best: {self.best_val_loss:.4f} at iter {self.best_iteration}). Worse count: {self.worse_count}/{self.patience}")
            if self.worse_count >= self.patience:
                raise RuntimeError(f"Early stopping triggered: Validation loss worsened {self.worse_count} consecutive times.")
            if val_loss > self.best_val_loss * 1.75 and iteration > 0:
                raise RuntimeError(f"Divergence detected: Validation loss ({val_loss:.4f}) surged >75% above best ({self.best_val_loss:.4f}).")


def run_preflight_checks():
    print("=" * 80)
    print("EVORAG TASK 12E: PREFLIGHT AUDIT & ENVIRONMENT CHECK")
    print("=" * 80)

    # 1. Environment
    print(f"Python executable: {sys.executable} ({platform.python_version()})")
    print(f"Machine architecture: {platform.machine()} (arm64: {platform.machine() == 'arm64'})")
    assert platform.machine() == "arm64", "Must run on native ARM64."
    
    metal_available = mx.metal.is_available()
    print(f"MLX Core: {mx.__version__} | MLX-LM: {mlx_lm.__version__} | Metal GPU: {metal_available}")
    assert metal_available, "Metal GPU acceleration is required."

    # 2. Disk Space
    free_disk_gib = get_free_disk_gib()
    print(f"Available free disk space: {free_disk_gib:.2f} GiB")
    if free_disk_gib < 5.0:
        raise RuntimeError(f"Insufficient disk space: {free_disk_gib:.2f} GiB available, at least 5.0 GiB required.")

    # 3. Dataset Integrity
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for fname in ["train.jsonl", "valid.jsonl", "internal_test.jsonl"]:
        path = DATASET_DIR / fname
        assert path.exists(), f"Dataset file missing: {path}"
        actual_hash = sha256_file(path)
        expected_hash = manifest["file_manifest"][fname]["sha256"]
        assert actual_hash == expected_hash, f"Hash mismatch for {fname}: got {actual_hash}, expected {expected_hash}"
        print(f"Dataset {fname:20s}: {actual_hash} (MATCH)")

    # 4. Base Model Immutability Verification
    base_hash = sha256_file(BASE_WEIGHT_FILE)
    assert base_hash == EXPECTED_BASE_SHA256, f"Base model hash mismatch: got {base_hash}, expected {EXPECTED_BASE_SHA256}"
    print(f"Base model safetensors:    {base_hash} (MATCH)")

    # 5. Smoke-Test Adapter Preservation
    assert SMOKE_ADAPTER_FILE.exists(), f"Smoke test adapter missing at {SMOKE_ADAPTER_FILE}"
    smoke_size_mb = SMOKE_ADAPTER_FILE.stat().st_size / (1024 * 1024)
    print(f"Smoke test adapter:        PRESERVED ({smoke_size_mb:.2f} MB)")

    # 6. Production Baseline Safeguards
    assert GGUF_MODEL.exists(), f"Production GGUF model missing at {GGUF_MODEL}"
    assert FROZEN_CSV.exists(), f"Frozen predictions CSV missing at {FROZEN_CSV}"
    frozen_hash = sha256_file(FROZEN_CSV)
    assert frozen_hash == EXPECTED_FROZEN_SHA256, f"Frozen predictions hash mismatch: {frozen_hash}"
    print(f"Frozen predictions CSV:    {frozen_hash} (MATCH)")
    print("Preflight audit completed successfully. All safeguards PASSED.\n")
    return free_disk_gib, base_hash


def main():
    initial_disk, initial_base_hash = run_preflight_checks()

    # Prepare adapter output directory
    if OUTPUT_ADAPTER_DIR.exists():
        print(f"Cleaning previous full training adapter directory: {OUTPUT_ADAPTER_DIR}")
        shutil.rmtree(OUTPUT_ADAPTER_DIR)
    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Base Model and Apply LoRA Configuration
    print("=" * 80)
    print("STEP 1: LOAD 4-BIT BASE MODEL & INITIALIZE LORA")
    print("=" * 80)
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    model, tokenizer = load(str(BASE_MODEL_DIR))
    load_duration = time.perf_counter() - load_start
    mem_after_load_gb = mx.get_peak_memory() / 1e9
    print(f"Base model loaded in {load_duration:.2f}s | Memory after load: {mem_after_load_gb:.2f} GB")

    lora_config = {"rank": 8, "scale": 16.0, "dropout": 0.0}
    num_lora_layers = 16
    print(f"Applying LoRA: rank={lora_config['rank']}, scale={lora_config['scale']}, num_layers={num_lora_layers}")
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=num_lora_layers,
        config=lora_config,
        use_dora=False,
    )
    print_trainable_parameters(model)

    trainable_params_before = dict(tree_flatten(model.trainable_parameters()))
    sample_param_name = list(trainable_params_before.keys())[0]
    initial_sample_weights = mx.array(trainable_params_before[sample_param_name])
    initial_sample_norm = mx.linalg.norm(initial_sample_weights).item()
    print(f"Initial sample adapter tensor '{sample_param_name}': L2-norm = {initial_sample_norm:.6f}")

    # 2. Load Datasets
    print("\n" + "=" * 80)
    print("STEP 2: LOAD TRAINING AND VALIDATION DATASETS")
    print("=" * 80)
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
    print(f"Training dataset:   {len(train_set)} examples (Train split)")
    print(f"Validation dataset: {len(valid_set)} examples (Validation split)")

    # 3. Configure Training
    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZE OPTIMIZER AND TRAINING HYPERPARAMETERS")
    print("=" * 80)
    learning_rate = 1.0e-5
    batch_size = 1
    epochs = 3
    total_iters = len(train_set) * epochs  # 182 * 3 = 546
    steps_per_report = 10
    steps_per_eval = 91                    # Every half-epoch (7 evaluation points)
    steps_per_save = len(train_set)        # 182: Exactly at each epoch boundary

    print(f"Optimizer:               Adam(lr={learning_rate})")
    print(f"Batch size:              {batch_size}")
    print(f"Total Epochs:            {epochs}")
    print(f"Total Iterations:        {total_iters}")
    print(f"Steps per report:        {steps_per_report}")
    print(f"Steps per eval:          {steps_per_eval} (every 0.5 epoch)")
    print(f"Steps per save:          {steps_per_save} (every 1.0 epoch)")
    print(f"Max sequence length:     1024")
    print(f"Prompt masking:          Enabled")

    optimizer = optim.Adam(learning_rate=learning_rate)

    adapter_file = OUTPUT_ADAPTER_DIR / "adapters.safetensors"
    config_dict = {
        "model": str(BASE_MODEL_DIR),
        "fine_tune_type": "lora",
        "num_layers": num_lora_layers,
        "batch_size": batch_size,
        "iters": total_iters,
        "learning_rate": learning_rate,
        "lora_parameters": lora_config,
        "mask_prompt": True,
        "max_seq_length": 1024,
    }
    save_config(config_dict, OUTPUT_ADAPTER_DIR / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=total_iters,
        val_batches=-1,                    # Evaluate all 39 validation samples
        steps_per_report=steps_per_report,
        steps_per_eval=steps_per_eval,
        steps_per_save=steps_per_save,
        adapter_file=adapter_file,
        max_seq_length=1024,
        grad_checkpoint=False,
        grad_accumulation_steps=1,
    )

    callback = FullTrainingCallback(model=model, adapter_dir=OUTPUT_ADAPTER_DIR, patience=5)

    # 4. Execute Full Training Loop
    print("\n" + "=" * 80)
    print("STEP 4: EXECUTE CONTROLLED QLORA TRAINING ON METAL GPU")
    print("=" * 80)
    train_start = time.perf_counter()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
        training_callback=callback,
    )
    train_duration = time.perf_counter() - train_start
    peak_training_mem_gb = mx.get_peak_memory() / 1e9

    print("\n" + "=" * 80)
    print("STEP 5: POST-TRAINING INTEGRITY & ARTIFACT CHECKS")
    print("=" * 80)
    print(f"Training completed in {train_duration:.2f}s ({train_duration/60:.2f} minutes).")
    print(f"Peak Metal memory during training: {peak_training_mem_gb:.2f} GB")

    # Check backpropagation weight update
    trainable_params_after = dict(tree_flatten(model.trainable_parameters()))
    final_sample_weights = mx.array(trainable_params_after[sample_param_name])
    weight_delta = mx.linalg.norm(final_sample_weights - initial_sample_weights).item()
    print(f"Sample adapter parameter '{sample_param_name}' weight delta: {weight_delta:.8f}")
    assert weight_delta > 0.0, "Adapter weights did not change after full training!"

    # Verify epoch checkpoints
    expected_checkpoints = [
        OUTPUT_ADAPTER_DIR / f"{182:07d}_adapters.safetensors",
        OUTPUT_ADAPTER_DIR / f"{364:07d}_adapters.safetensors",
        OUTPUT_ADAPTER_DIR / f"{546:07d}_adapters.safetensors",
        OUTPUT_ADAPTER_DIR / "adapters.safetensors",
        OUTPUT_ADAPTER_DIR / "best_adapters.safetensors",
    ]
    for ckpt in expected_checkpoints:
        assert ckpt.exists(), f"Expected checkpoint missing: {ckpt}"
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"Checkpoint verified: {ckpt.name:30s} ({size_mb:.2f} MB)")

    # If best_adapters.safetensors wasn't saved because val loss only decreased at final step:
    best_adapter_path = OUTPUT_ADAPTER_DIR / "best_adapters.safetensors"
    best_info = {
        "best_iteration": callback.best_iteration,
        "best_val_loss": callback.best_val_loss,
        "best_adapter_file": str(best_adapter_path),
        "best_adapter_sha256": sha256_file(best_adapter_path),
    }
    with open(OUTPUT_ADAPTER_DIR / "best_checkpoint_info.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    print(f"Best checkpoint info saved: Best Val Loss = {callback.best_val_loss:.4f} at Iteration {callback.best_iteration}")

    # Verify base model weights immutability
    post_base_hash = sha256_file(BASE_WEIGHT_FILE)
    assert post_base_hash == initial_base_hash, f"Base model weights mutated! Before: {initial_base_hash}, After: {post_base_hash}"
    print(f"Base model weights 100% UNCHANGED: {post_base_hash}")

    # Verify frozen benchmark predictions immutability
    post_frozen_hash = sha256_file(FROZEN_CSV)
    assert post_frozen_hash == EXPECTED_FROZEN_SHA256, f"Frozen benchmark predictions mutated!"
    print(f"Frozen benchmark predictions 100% UNCHANGED: {post_frozen_hash}")

    # Check smoke test adapter preservation
    assert SMOKE_ADAPTER_FILE.exists(), "Smoke test adapter missing after training!"
    print(f"Smoke test adapter preserved: {SMOKE_ADAPTER_FILE}")

    final_disk = get_free_disk_gib()
    print(f"Disk space: Initial {initial_disk:.2f} GiB -> Final {final_disk:.2f} GiB (Net used: {initial_disk - final_disk:.2f} GiB)")

    # Save training metrics
    metrics = {
        "task": "EvoRAG Task 12E Controlled Full DBMS QLoRA Training",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "arch": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__,
            "initial_free_disk_gib": round(initial_disk, 2),
            "final_free_disk_gib": round(final_disk, 2),
        },
        "base_model": {
            "path": str(BASE_MODEL_DIR),
            "sha256": post_base_hash,
            "immutability_verified": True,
        },
        "training_configuration": {
            "num_layers": num_lora_layers,
            "rank": lora_config["rank"],
            "scale": lora_config["scale"],
            "dropout": lora_config["dropout"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "total_iterations": total_iters,
            "max_seq_length": 1024,
            "mask_prompt": True,
            "steps_per_report": steps_per_report,
            "steps_per_eval": steps_per_eval,
            "steps_per_save": steps_per_save,
        },
        "training_results": {
            "duration_seconds": round(train_duration, 2),
            "duration_minutes": round(train_duration / 60, 2),
            "peak_memory_gb": round(peak_training_mem_gb, 2),
            "initial_train_loss": round(callback.train_reports[0]["train_loss"], 4),
            "final_train_loss": round(callback.train_reports[-1]["train_loss"], 4),
            "best_val_loss": round(callback.best_val_loss, 4),
            "best_iteration": callback.best_iteration,
            "weight_delta_sample": round(weight_delta, 8),
            "train_reports": callback.train_reports,
            "val_reports": callback.val_reports,
        },
        "artifacts": {
            "adapter_dir": str(OUTPUT_ADAPTER_DIR),
            "checkpoints": [ckpt.name for ckpt in expected_checkpoints],
            "best_checkpoint": best_info,
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull training metrics saved to: {METRICS_FILE}")
    print("=" * 80)
    print("TRAINING LOOP SUCCESSFULLY EXECUTED")
    print("=" * 80)


if __name__ == "__main__":
    main()
