"""Experiment configuration.

Defaults live here; every field can be overridden from `.env` (or the real
environment) with a `KRONA_` prefix, e.g. `KRONA_MODEL`, `KRONA_STEPS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` once, at import time. `override=False` means the real environment wins,
# which is what we want on RunPod (pod env vars, `docker run -e`, etc.).
load_dotenv(override=False)

ENV_PREFIX = "KRONA_"


def _default_out_dir() -> str:
    """RunPod writes to `/workspace` (survives a pod restart); elsewhere `outputs/`."""
    if Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
        return "/workspace/outputs"
    return "outputs"


@dataclass
class Config:
    """Everything that controls a run. One instance is passed around explicitly."""

    # --- model ---
    model: str = "unsloth/gemma-3-1b-it"

    # MUST stay False for KronA. PEFT's LoKrLayer does
    #     weight.reshape(base_layer.weight.shape)
    # and on a bitsandbytes 4-bit layer `weight.shape` is the *nibble-packed* storage,
    # so the reshape fails with
    #     RuntimeError: shape '[589824, 1]' is invalid for input of size 1179648
    # (589824 is exactly half of 1179648, i.e. the 4-bit packing).
    # LoKr is therefore NOT QLoRA-compatible. 16-bit is required, and BOTH arms use it
    # so the comparison stays fair. See docs/NOTES.md.
    load_in_4bit: bool = False

    # --- data ---
    dataset: str = "databricks/databricks-dolly-15k"
    max_samples: int = 1000
    eval_size: int = 64
    max_seq: int = 512

    # --- training (identical for both adapters) ---
    steps: int = 60
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    eval_every: int = 20
    warmup_steps: int = 5
    logging_steps: int = 5

    # --- adapter (same rank/alpha for both methods) ---
    rank: int = 16
    alpha: int = 32

    # --- outputs / misc ---
    out_dir: str = ""
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.out_dir:
            self.out_dir = _default_out_dir()

    # ---------------------------------------------------------------- helpers
    @property
    def adapters(self) -> list[str]:
        """The two methods, in run order."""
        return ["lora", "krona"]

    def adapter_dir(self, kind: str) -> Path:
        return Path(self.out_dir) / f"adapter_{kind}"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from class defaults + `KRONA_*` environment variables."""
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"{ENV_PREFIX}{f.name.upper()}")
            if raw is None or raw == "":
                continue
            # Coerce to the field's declared type via its default.
            default = getattr(cfg, f.name)
            try:
                if isinstance(default, bool):
                    value = raw.strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    value = int(raw)
                elif isinstance(default, float):
                    value = float(raw)
                else:
                    value = raw
            except ValueError as exc:
                raise ValueError(
                    f"Bad value for {ENV_PREFIX}{f.name.upper()}={raw!r}: {exc}"
                ) from exc
            setattr(cfg, f.name, value)
        return cfg


if __name__ == "__main__":  # `python -m krona.config` prints the effective config
    c = Config.from_env()
    width = max(len(f.name) for f in fields(Config))
    for f in fields(Config):
        print(f"{f.name:<{width}} = {getattr(c, f.name)!r}")
