# Plan — Fine-tuning LoRA vs KronA on Gemma 4 E2B (Colab)

> Status: plan approved, **no code written yet.**
> Goal: fine-tune `google/gemma-4-E2B-it` for chat **twice** — once with **LoRA**,
> once with **KronA** — then compare. Runs in a **Colab** notebook (free T4).

---

## Decisions

| | |
|---|---|
| Model | `google/gemma-4-E2B-it` (4-bit: `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`) |
| Dataset | `databricks/databricks-dolly-15k` (capped) |
| Task | Chat / instruction SFT |
| Methods | **LoRA** (`LoraConfig`) vs **KronA** (`LoKrConfig`) |
| Compute | Colab T4 16 GB, 4-bit + fp16 |
| Hub push | No |

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
- [X] 2. Accept the Gemma license: huggingface.co/`google/gemma-4-E2B-it` *(you)*
- [X] 3. Create a Hugging Face **read token** *(you)*
- [X] 4. Create `README.md`
- [X] 5. ~~Create `requirements.txt`~~ — dropped: deps are pinned in the notebook instead
- [X] 6. Create `.gitignore`
- [ ] 7. Create `notebooks/lora_krona_gemma4_e2b.ipynb` — **main deliverable**
- [ ] 8. Cell: install libs + check the GPU
- [ ] 9. Cell: log in to Hugging Face
- [ ] 10. Cell: load the 4-bit model + tokenizer
- [ ] 11. Cell: load and format the dataset
- [ ] 12. Cell: **train A — LoRA**
- [ ] 13. Cell: **train B — KronA**
- [ ] 14. Cell: compare (params, VRAM, time, loss, samples)
- [ ] 15. Cell: save both adapters
- [ ] 16. Validate end-to-end on Colab and fix issues
- [ ] 17. Send me any error you hit *(you)*

---

## Rules

- Both runs use the **same data, seed, steps, learning rate, batch size and target modules** — only the adapter type differs.
- Cap the dataset with one easy knob (`MAX_SAMPLES`) so a run stays fast.
- Dependencies are **pinned inside the notebook** (setup cell). No `requirements.txt`.
- **Notebook-only:** no helper `.py` files, no `configs/`.
