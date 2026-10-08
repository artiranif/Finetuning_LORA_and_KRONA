"""Model loading and the shared training loop."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, fields

import torch
from trl import SFTConfig, SFTTrainer

from .adapters import attach_adapter
from .config import Config

PROMPTS = [
    "Explain what a Kronecker product is in one sentence.",
    "Give me 3 tips to speed up a PyTorch training loop.",
    "Write a haiku about fine-tuning a language model.",
]


@dataclass
class Result:
    """Everything we report about one adapter run."""

    method: str
    trainable: int
    total: int
    eval_loss: float
    perplexity: float
    seconds: float
    peak_vram_gb: float
    samples: list[str]

    @property
    def trainable_pct(self) -> float:
        return 100.0 * self.trainable / self.total

    def to_row(self) -> dict:
        return {
            "method": self.method.upper(),
            "trainable": self.trainable,
            "trainable_%": round(self.trainable_pct, 3),
            "eval_loss": round(self.eval_loss, 4),
            "perplexity": round(self.perplexity, 2),
            "time_s": round(self.seconds),
            "peak_VRAM_GB": round(self.peak_vram_gb, 2),
        }


def load_base(cfg: Config):
    """Load a fresh base model + tokenizer. Unsloth does the memory-safe setup itself
    (no fp32 upcast of every non-quantised parameter, which is what OOM'ed the earlier
    plain transformers + peft version on a T4)."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model,
        load_in_4bit=cfg.load_in_4bit,
        max_seq_length=cfg.max_seq,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    model.config.use_cache = False
    tokenizer.padding_side = "right"
    return model, tokenizer


def sft_config(**kwargs) -> SFTConfig:
    """Build an SFTConfig, dropping kwargs this TRL version does not accept.

    Keeps us compatible across TRL releases (`eval_strategy` vs `evaluation_strategy`,
    `max_length` vs `max_seq_length`, ...) without hard-coding a version.
    """
    accepted = {f.name for f in fields(SFTConfig)}
    if "evaluation_strategy" in accepted and "eval_strategy" not in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in accepted})


def generate(model, tokenizer, prompts=PROMPTS, max_new_tokens: int = 100) -> list[str]:
    """Greedy generations for the side-by-side comparison."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device
    outputs = []
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        outputs.append(
            tokenizer.decode(generated[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        )
    return outputs


def train_one(kind: str, cfg: Config, train_ds, eval_ds, tokenizer=None) -> Result:
    """Train a single adapter from a fresh base model and return its Result."""
    from unsloth import FastLanguageModel

    model, tokenizer = load_base(cfg)
    model, trainable, total = attach_adapter(model, kind, cfg)
    print(f"\n=== {kind.upper()} — {trainable:,}/{total:,} trainable "
          f"({100.0 * trainable / total:.3f}%) ===")

    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=sft_config(
            output_dir=f"{cfg.out_dir}/run_{kind}",
            max_steps=cfg.steps,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.lr,
            lr_scheduler_type="cosine",
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.eval_every,
            optim="adamw_8bit",
            seed=cfg.seed,
            report_to="none",
            # No fp16: Unsloth forces Gemma to float32 (fp16 overflows Gemma), and we
            # should not fight that. bf16 only where the GPU actually supports it.
            bf16=torch.cuda.is_bf16_supported(),
            max_length=cfg.max_seq,
            packing=False,
        ),
    )
    trainer.train()

    seconds = time.time() - started
    peak_vram = torch.cuda.max_memory_reserved() / 1e9
    eval_loss = float(trainer.evaluate()["eval_loss"])
    samples = generate(model, tokenizer)

    out_dir = cfg.adapter_dir(kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved adapter -> {out_dir}")

    # Free VRAM before the next arm.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return Result(
        method=kind,
        trainable=trainable,
        total=total,
        eval_loss=eval_loss,
        perplexity=float(torch.exp(torch.tensor(eval_loss))),
        seconds=seconds,
        peak_vram_gb=peak_vram,
        samples=samples,
    )


def run_all(cfg: Config, train_ds, eval_ds) -> dict[str, Result]:
    """Run both arms, LoRA then KronA, each from a fresh base model."""
    return {kind: train_one(kind, cfg, train_ds, eval_ds) for kind in cfg.adapters}
