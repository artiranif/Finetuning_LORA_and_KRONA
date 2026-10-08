"""Runtime environment detection, dependency checks and Hugging Face login."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def is_colab() -> bool:
    return "google.colab" in os.sys.modules or Path("/content").is_dir()


def is_runpod() -> bool:
    """RunPod sets RUNPOD_* env vars and mounts /workspace."""
    return any(k.startswith("RUNPOD_") for k in os.environ) or Path("/workspace").is_dir()


def runtime_name() -> str:
    if is_runpod():
        return "RunPod"
    if is_colab():
        return "Colab"
    return "local"


def gpu_report() -> dict:
    """Small dict describing the GPU, or `{"gpu": None}` when there is none."""
    import torch

    if not torch.cuda.is_available():
        return {"gpu": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1e9, 2),
        "capability": f"{props.major}.{props.minor}",
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runtime": runtime_name(),
    }


def require_gpu() -> None:
    """Fail early and clearly instead of deep inside the trainer."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible.\n"
            "  - RunPod: pick a GPU pod (not a CPU pod).\n"
            "  - Colab : Runtime -> Change runtime type -> GPU."
        )


def hf_login() -> bool:
    """Log in to the Hugging Face Hub if a token is available.

    Order of precedence:
      1. `HF_TOKEN` in the environment / `.env`
      2. Colab secret named `HF_TOKEN`
      3. interactive prompt

    Returns True if logged in. Returns False (without prompting) when the token is
    absent and `HF_ALLOW_ANONYMOUS` is set -- handy for the public Gemma 3 repos,
    which download fine anonymously.
    """
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN") or None

    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None

    if not token:
        if os.environ.get("HF_ALLOW_ANONYMOUS", "").lower() in ("1", "true", "yes"):
            print("No HF_TOKEN; continuing anonymously (public repos only).")
            return False
        from getpass import getpass

        token = getpass("Hugging Face token (read), or Enter to skip: ") or None

    if not token:
        print("No token supplied; continuing anonymously (public repos only).")
        return False

    login(token=token, add_to_git_credential=False)
    print("Logged in to Hugging Face.")
    return True


def check_dependencies() -> None:
    """Verify the training stack imports, with an actionable message if not."""
    missing = []
    for mod in ("torch", "datasets", "peft", "trl", "unsloth"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            "Missing dependencies: " + ", ".join(missing) + "\n"
            "  pip install -r requirements.txt\n"
            "On an unusual CUDA/PyTorch combo, let Unsloth pick the build tag:\n"
            "  wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/"
            "unsloth/_auto_install.py | python -"
        )


def print_environment() -> None:
    info = gpu_report()
    if info.get("gpu") is None:
        print(f"runtime: {runtime_name()} | GPU: none")
        return
    print(
        f"runtime: {info['runtime']} | GPU: {info['gpu']} "
        f"({info['vram_gb']} GB, cc {info['capability']}) | bf16: {info['bf16']} | "
        f"torch {info['torch']} / CUDA {info['cuda']}"
    )
