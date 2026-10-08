#!/usr/bin/env python
"""Run the whole LoRA-vs-KronA experiment.

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


def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for f in fields(Config):
        parser.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, default=None)
    parser.add_argument("--print-config", action="store_true", help="print config and exit")
    parser.add_argument("--skip-login", action="store_true", help="do not log in to HF")
    args = parser.parse_args()

    cfg = Config.from_env()
    for f in fields(Config):
        value = getattr(args, f.name)
        if value is None:
            continue
        # argparse yields strings; coerce to the field's type via its default.
        default = getattr(cfg, f.name)
        if isinstance(default, bool):
            value = str(value).strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        setattr(cfg, f.name, value)
    return cfg, args


def main() -> int:
    cfg, args = parse_args()

    if args.print_config:
        width = max(len(f.name) for f in fields(Config))
        for f in fields(Config):
            print(f"{f.name:<{width}} = {getattr(cfg, f.name)!r}")
        return 0

    if args.skip_login:
        import os

        os.environ["HF_ALLOW_ANONYMOUS"] = "1"

    from krona.pipeline import run

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

