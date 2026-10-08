#!/usr/bin/env python
"""Run the whole LoRA-vs-KronA experiment. Works on Colab, RunPod or locally.

    python run.py                      # defaults + .env
    python run.py --steps 200          # override a knob
    python run.py --print-config       # show the effective config and exit

Every `--flag` maps to a Config field; the same fields can be set via `KRONA_*`
environment variables (see `.env.mock`).
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from dotenv import load_dotenv

load_dotenv(override=False)

from krona.config import Config  # noqa: E402
from krona.env import check_dependencies, hf_login, print_environment, require_gpu  # noqa: E402


def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        parser.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, default=None)
    parser.add_argument("--print-config", action="store_true", help="print config and exit")
    parser.add_argument("--skip-login", action="store_true", help="do not log in to Hugging Face")
    args = parser.parse_args()

    cfg = Config.from_env()
    for f in fields(Config):
        value = getattr(args, f.name)
        if value is None:
            continue
        default = getattr(cfg, f.name)
        # argparse gives strings; coerce to the field's type via its default.
        if isinstance(default, bool):
            value = str(value).strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int) and not isinstance(default, bool):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        setattr(cfg, f.name, value)
    if not cfg.out_dir:
        Config.__post_init__(cfg)
    return cfg, args


def main() -> int:
    cfg, args = parse_args()

    if args.print_config:
        width = max(len(f.name) for f in fields(Config))
        for f in fields(Config):
            print(f"{f.name:<{width}} = {getattr(cfg, f.name)!r}")
        return 0

    print_environment()
    require_gpu()
    check_dependencies()
    if not args.skip_login:
        try:
            hf_login()
        except Exception as exc:  # a bad token should not be fatal for public repos
            print(f"Hugging Face login skipped ({type(exc).__name__}: {exc})")

    from krona.data import build_dataset
    from krona.report import summarize
    from krona.train import load_base, run_all

    # A throwaway load just to get the tokenizer for chat-template rendering,
    # then release it so the two training arms start from a clean card.
    model, tokenizer = load_base(cfg)
    del model
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    train_ds, eval_ds = build_dataset(cfg, tokenizer)
    print(f"train: {len(train_ds)} rows | eval: {len(eval_ds)} rows")
    print(f"example:\n{train_ds[0]['text'][:300]}")

    results = run_all(cfg, train_ds, eval_ds)
    summarize(results, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
