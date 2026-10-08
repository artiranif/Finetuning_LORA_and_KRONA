"""Comparison table, side-by-side samples and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .train import PROMPTS, Result


def print_table(results: dict[str, Result]) -> None:
    """Print the head-to-head table without needing pandas."""
    import pandas as pd

    frame = pd.DataFrame([r.to_row() for r in results.values()])
    try:
        from IPython.display import display

        display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def print_samples(results: dict[str, Result], prompts=PROMPTS) -> None:
    for i, prompt in enumerate(prompts):
        print("\n" + "-" * 70 + f"\nQ: {prompt}")
        for kind, result in results.items():
            answer = result.samples[i] if i < len(result.samples) else "<missing>"
            print(f"[{kind.upper():5}] {answer}")


def save_results(results: dict[str, Result], cfg: Config) -> Path:
    """Write metrics to `<out_dir>/results.json` so runs can be diffed later.

    This is the artifact to keep: adapters and `results.json` are small, the pod is not.
    """
    out = Path(cfg.out_dir) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "model": cfg.model,
            "load_in_4bit": cfg.load_in_4bit,
            "dataset": cfg.dataset,
            "max_samples": cfg.max_samples,
            "eval_size": cfg.eval_size,
            "max_seq": cfg.max_seq,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "lr": cfg.lr,
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "seed": cfg.seed,
        },
        "results": {k: r.to_row() for k, r in results.items()},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved metrics -> {out}")
    return out


def summarize(results: dict[str, Result], cfg: Config) -> None:
    """Everything the notebook/CLI prints at the end."""
    print("\n=== LoRA vs KronA ===")
    print_table(results)
    print("\n=== sample outputs ===")
    print_samples(results)
    save_results(results, cfg)
