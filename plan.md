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

> **Notebook-only:** all the code lives in the notebook — no `src/` or `configs/` files.

```text
plan.md  README.md  .gitignore
notebooks/lora_krona_gemma4_e2b.ipynb   <- main deliverable
memories/memory.md
```

---

## Checklist

- [X] 1. Colab → `Runtime → Change runtime type → T4 GPU` *(you)*
- [X] 2. ~~Accept the Gemma license~~ — not needed any more: `unsloth/gemma-3-1b-it` is public
- [X] 3. Create a Hugging Face **read token** *(you)*
- [X] 4. Create `README.md`
- [X] 5. ~~Create `requirements.txt`~~ — dropped: the notebook installs Unsloth itself
- [X] 6. Create `.gitignore`
- [X] 7. Create `notebooks/lora_krona_gemma4_e2b.ipynb` — **main deliverable**
- [X] 8. Cell 1: install Unsloth + log in to Hugging Face
- [X] 9. Cell 2: config + load the 4-bit model with Unsloth
- [X] 10. Cell 3: build the chat dataset
- [X] 11. Cell 4: **train A — LoRA**, then **train B — KronA** (shared loop)
- [X] 12. Cell 5: compare (params, VRAM, time, loss, perplexity, samples) + save adapters
- [X] 13. Full Unsloth rewrite: 5 code cells instead of 9, plain `peft` adapters for fairness
- [ ] 14. Validate end-to-end on Colab and fix issues *(you)*
- [ ] 15. Send me any error you hit *(you)*

---

## Rules

- Both runs use the **same data, seed, steps, learning rate, batch size and target modules** — only the adapter type differs.
- Cap the dataset with one easy knob (`max_samples`) so a run stays fast.
- Dependencies are installed **inside the notebook** (Unsloth pins its own versions). No `requirements.txt`.
- **Notebook-only:** no helper `.py` files, no `configs/`.
- **Unsloth** loads/patches the model; adapters are attached with plain `peft` because Unsloth has no `LoKr` support.
