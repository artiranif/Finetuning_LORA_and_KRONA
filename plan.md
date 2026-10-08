# Plan — Fine-tuning LoRA vs KronA with Unsloth (Colab)

> Status: **rewritten around Unsloth** (2026-10-08). Notebook exists; not executed on Colab yet.
> Goal: fine-tune a small chat model **twice** — once with **LoRA**, once with **KronA** —
> then compare. Runs in a **Colab** notebook (free T4), fast.

---

## Decisions

| | |
|---|---|
| Engine | **Unsloth** (`FastLanguageModel.from_pretrained`) for load + patching |
| Model | `unsloth/gemma-3-1b-it` — small, text-only, **public**, fast |
| Dataset | `databricks/databricks-dolly-15k` (capped) |
| Task | Chat / instruction SFT |
| Methods | **LoRA** (`LoraConfig`) vs **KronA** (`LoKrConfig`), attached via plain `peft` |
| Compute | Colab T4 16 GB, 4-bit + fp16 |
| Hub push | No |

> Earlier target was `google/gemma-4-E2B-it`, dropped for speed/reliability (gated repo, needs
> transformers ≥ 5.5, tight VRAM). Set `CFG["model"] = "unsloth/gemma-4-E2B-it"` to go back.

> **Fairness:** Unsloth has **no `LoKr` support**, so both adapters are attached by hand with
> `peft.get_peft_model()` on the same Unsloth-loaded base. Both arms get identical speedups
> and only the adapter type differs.

> **KRONA = KronA** = the Kronecker adapter, arXiv:2212.10650. In HuggingFace PEFT it is the **`LoKr`** adapter.

---

## Files to create

> **Now a real Python project** (2026-10-08): the logic lives in the `krona/` package so the
> notebook, the CLI and future runs share one code path. The notebook is a thin wrapper.

```text
plan.md  README.md  CHEATSHEET.md  requirements.txt  .gitignore  .env.mock
docs/NOTES.md                             gotchas & decisions
krona/{config,env,data,adapters,train,report,pipeline}.py
run.py                                    CLI entry point
compare.py                                diff two results.json
scripts/setup_runpod.sh  scripts/pack_outputs.sh
lora_krona_gemma4_e2b.ipynb               minimal 2-cell driver (repo root)
memories/memory.md
```

---

## Checklist

- [X] 1. GPU runtime (Colab T4 or a RunPod GPU pod)
- [X] 2. ~~Accept the Gemma license~~ — not needed: `unsloth/gemma-3-1b-it` is public
- [X] 3. HF read token — now **optional** (`HF_ALLOW_ANONYMOUS=1` skips it)
- [X] 4. `README.md`, `CHEATSHEET.md`, `docs/NOTES.md`
- [X] 5. `requirements.txt` (unsloth, datasets, hf_transfer, python-dotenv, pandas)
- [X] 6. `.gitignore`, `.env.mock`
- [X] 7. `krona/` package: config / env / data / adapters / train / report
- [X] 8. `run.py` CLI (every Config field is a flag; `KRONA_*` env overrides)
- [X] 9. `compare.py` — diff two `results.json` runs
- [X] 10. RunPod: runtime detection, `/workspace/outputs`, setup + pack scripts
- [X] 11. Notebook reduced to a thin wrapper (no duplicated logic)
- [X] 12. Fix `LoraConfig` vs `LoKrConfig` kwarg mismatch, drop `fp16=True`
- [X] 13. Switch to 16-bit (LoKr is not QLoRA-compatible)
- [X] 14. Notebook moved to the repo ROOT and cut to 2 cells / 12 lines; `krona/pipeline.py`
- [ ] 15. Validate end-to-end on Colab/RunPod and fix issues *(you)*
- [ ] 16. Send me any error you hit *(you)*

---

## Rules

- Both runs use the **same data, seed, steps, learning rate, batch size and target modules** — only the adapter type differs.
- Cap the dataset with one easy knob (`max_samples`) so a run stays fast.
- **No logic in the notebook** — it calls `krona.pipeline.run()`. `python run.py` does the same thing.
- **`load_in_4bit` must stay `False`** — LoKr is not QLoRA-compatible (see `docs/NOTES.md`).
- **Unsloth** loads/patches the model; adapters are attached with plain `peft` because Unsloth has no `LoKr` support.
- Never hardcode `fp16=True` for Gemma (it overflows; Unsloth forces float32).
