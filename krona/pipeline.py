"""The whole experiment, in one call.

`run.py` and the notebook both go through `run()`, so there is exactly one code path
and no duplicated logic to keep in sync.
"""

from __future__ import annotations

from .config import Config


def run(cfg: Config | None = None) -> dict:
    """Environment check -> dataset -> LoRA + KronA -> comparison.

    Returns {adapter_kind: Result}. Artifacts land in `cfg.out_dir`.
    """
    import gc

    import torch

    from .data import build_dataset
    from .env import check_dependencies, hf_login, print_environment, require_gpu
    from .report import summarize
    from .train import load_base, run_all

    cfg = cfg or Config.from_env()

    print_environment()
    require_gpu()
    check_dependencies()
    try:
        hf_login()                     # optional: public repos need no token
    except Exception as exc:
        print(f"Hugging Face login skipped ({type(exc).__name__}: {exc})")

    # Only the tokenizer is needed here, to render the chats with the model's own
    # template. Release the weights again so both arms start from a clean card.
    model, tokenizer = load_base(cfg)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_ds, eval_ds = build_dataset(cfg, tokenizer)
    print(f"train: {len(train_ds)} rows | eval: {len(eval_ds)} rows")

    results = run_all(cfg, train_ds, eval_ds)
    summarize(results, cfg)
    return results
