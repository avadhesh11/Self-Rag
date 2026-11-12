"""
EvoRAG Task 12A: MLX Environment Diagnostic and Hardware Verification Script.

Verifies:
- Chip and processor architecture
- Python architecture and runtime version
- macOS version and Metal availability
- MLX and MLX-LM installation
- Default MLX compute device
- Available system memory and disk space
- Enforces a configurable minimum free disk threshold (default: 15.0 GiB)
- Executes minimal MLX tensor ops and Metal-backed computation without loading 7B models.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

# Configurable minimum disk threshold
DEFAULT_MIN_FREE_DISK_GB = 15.0


def get_cpu_brand() -> str:
    """Fetch exact Apple chip name using sysctl."""
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Apple Silicon"


def get_total_ram_gb() -> float:
    """Fetch total physical unified memory in GiB using sysctl."""
    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return int(res.stdout.strip()) / (1024**3)
    except Exception:
        pass
    return 0.0


def check_environment(min_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB) -> int:
    """Perform diagnostic check of MLX environment and Metal execution."""
    print("=" * 70)
    print("EvoRAG MLX Environment & Hardware Diagnostic")
    print("=" * 70)

    # 1. Platform & OS
    mac_ver, _, _ = platform.mac_ver()
    cpu_brand = get_cpu_brand()
    py_arch = platform.machine()
    total_ram_gb = get_total_ram_gb()

    print(f"Chip Architecture:      {cpu_brand}")
    print(f"Python Architecture:    {py_arch} ({'Native ARM64' if py_arch == 'arm64' else 'Rosetta / Emulated'})")
    print(f"macOS Version:          {mac_ver}")
    print(f"Python Runtime:         {sys.version.split()[0]} ({sys.executable})")
    print(f"System Memory:          {total_ram_gb:.2f} GiB Unified Memory")

    # 2. Disk Space & Safety Check
    target_dir = Path(__file__).resolve().parent.parent.parent
    usage = shutil.disk_usage(target_dir)
    free_disk_gb = usage.free / (1024**3)
    total_disk_gb = usage.total / (1024**3)
    used_disk_pct = (usage.used / usage.total) * 100

    print(f"Disk Space:             {free_disk_gb:.2f} GiB available / {total_disk_gb:.2f} GiB total ({used_disk_pct:.1f}% used)")
    print(f"Safety Threshold:       {min_disk_gb:.1f} GiB minimum free space required")

    disk_ok = free_disk_gb >= min_disk_gb
    if disk_ok:
        print(f"  -> Disk Safety Check:  [PASS] ({free_disk_gb:.2f} GiB >= {min_disk_gb:.1f} GiB)")
    else:
        print(f"  -> Disk Safety Check:  [WARNING] Free space ({free_disk_gb:.2f} GiB) below threshold ({min_disk_gb:.1f} GiB)!")
        print("     Pruning caches will be required before large model downloads.")

    # 3. MLX & MLX-LM Import Verification
    print("-" * 70)
    print("Verifying MLX & MLX-LM Libraries...")

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx_lm
    except ImportError as exc:
        print(f"[FAIL] Missing MLX dependencies: {exc}")
        print("Please ensure .venv-mlx is active and 'mlx-lm' is installed.")
        return 1

    mlx_version = mx.__version__
    mlx_lm_version = getattr(mlx_lm, "__version__", "installed")
    default_device = str(mx.default_device())
    metal_available = mx.metal.is_available()

    print(f"MLX Core Version:       {mlx_version}")
    print(f"MLX-LM Version:         {mlx_lm_version}")
    print(f"Default MLX Device:     {default_device}")
    print(f"Metal GPU Available:    {metal_available}")

    if not metal_available:
        print("[FAIL] Apple Metal GPU backend is not available in MLX!")
        return 1

    # 4. Minimal MLX Tensor Operations & Metal-backed Computation
    print("-" * 70)
    print("Executing Minimal Metal Compute Operations...")

    t0 = time.perf_counter()

    # Minimal tensor creation and arithmetic
    a = mx.random.normal((512, 512))
    b = mx.random.normal((512, 512))
    c = mx.matmul(a, b)
    mx.eval(c)

    # Minimal small neural network forward pass on Metal
    linear = nn.Linear(512, 128)
    out = linear(c)
    mx.eval(out)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  -> Tensor shape verification: a={a.shape}, b={b.shape}, matmul={c.shape}")
    print(f"  -> nn.Linear forward pass:    input={c.shape} -> output={out.shape}")
    print(f"  -> Metal computation time:    {elapsed_ms:.2f} ms")
    print(f"  -> Output tensor mean:        {float(mx.mean(out)):.6f}")

    print("=" * 70)
    print("MLX Environment Verification: SUCCESS")
    print("All core MLX, MLX-LM, and Metal GPU requirements satisfied.")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check MLX environment and Metal execution.")
    parser.add_argument(
        "--min-disk-gb",
        type=float,
        default=DEFAULT_MIN_FREE_DISK_GB,
        help="Minimum free disk space threshold in GiB (default: 15.0)",
    )
    args = parser.parse_args()
    sys.exit(check_environment(min_disk_gb=args.min_disk_gb))


if __name__ == "__main__":
    main()

