"""Dataset preparation: dolly-15k -> a chat-rendered `text` column."""

from __future__ import annotations

from datasets import Dataset, load_dataset

from .config import Config


def build_messages(example: dict) -> list[dict]:
    """Turn one dolly row into user/assistant chat turns."""
    user = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    if context:
        user = f"{user}\n\n{context}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": (example.get("response") or "").strip()},
    ]


def build_dataset(cfg: Config, tokenizer) -> tuple[Dataset, Dataset]:
    """Return `(train, eval)`.

    Rows are pre-rendered with the tokenizer's own chat template into a plain `text`
    column. A pre-rendered string is the format every TRL version accepts, whereas a
    `messages` column depends on version-specific auto-detection.
    """
    raw = (
        load_dataset(cfg.dataset, split="train")
        .shuffle(seed=cfg.seed)
        .select(range(min(cfg.max_samples, len(load_dataset(cfg.dataset, split="train")))))
    )

    raw = raw.filter(lambda e: (e.get("response") or "").strip())

    def to_text(example: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(build_messages(example), tokenize=False)}

    ds = raw.map(to_text, remove_columns=raw.column_names)

    n = len(ds)
    if n <= cfg.eval_size + 1:
        raise ValueError(
            f"Not enough rows ({n}) for eval_size={cfg.eval_size}. "
            "Raise max_samples or lower eval_size."
        )
    cut = n - cfg.eval_size
    return ds.select(range(cut)), ds.select(range(cut, n))
