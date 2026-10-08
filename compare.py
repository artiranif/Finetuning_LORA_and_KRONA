#!/usr/bin/env python
"""Compare the metrics of two (or more) `results.json` files from previous runs.

    python compare.py outputs/results.json
    python compare.py run_a/results.json run_b/results.json

Useful for diffing a Colab run against a RunPod run, or one rank against another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    label = data.get("config", {}).get("model", "?")
    rows = []
    for method, row in data.get("results", {}).items():
        rows.append({"run": f"{Path(path).parent.name or '.'} ({label})", **row})
    return {"rows": rows, "config": data.get("config", {})}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    rows: list[dict] = []
    for path in argv:
        if not Path(path).exists():
            print(f"missing file: {path}")
            return 1
        rows.extend(load(path)["rows"])

    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        print(frame.to_string(index=False))
    except ImportError:
        for row in rows:
            print(row)

    # Show which config keys differ between runs, so a comparison is never apples-to-oranges.
    configs = [load(p)["config"] for p in argv]
    if len(configs) > 1:
        keys = sorted(set().union(*(c.keys() for c in configs)))
        diffs = {
            k: [c.get(k) for c in configs]
            for k in keys
            if len({json.dumps(c.get(k), sort_keys=True) for c in configs}) > 1
        }
        print("\nconfig differences between runs:")
        print(json.dumps(diffs, indent=2) if diffs else "  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
