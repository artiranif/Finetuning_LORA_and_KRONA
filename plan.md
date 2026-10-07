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

```text
plan.md  README.md  requirements.txt  .gitignore
notebooks/lora_krona_gemma4_e2b.ipynb   <- main deliverable
src/  data.py  train.py  evaluate.py
configs/  lora.yaml  lokr.yaml
memories/memory.md
```

---

## Checklist

- [ ] 1. Colab → `Runtime → Change runtime type → T4 GPU` *(you)*
- [ ] 2. Accept the Gemma license: huggingface.co/`google/gemma-4-E2B-it` *(you)*
- [ ] 3. Create a Hugging Face **read token** *(you)*
- [ ] 4. Create `README.md`
- [ ] 5. Create `requirements.txt`
- [ ] 6. Create `.gitignore`
- [ ] 7. Create `src/data.py` — load + format the dataset
- [ ] 8. Create `src/train.py` — build + train one adapter
- [ ] 9. Create `src/evaluate.py` — loss / perplexity / samples
- [ ] 10. Create `configs/lora.yaml`
- [ ] 11. Create `configs/lokr.yaml`
- [ ] 12. Create `notebooks/lora_krona_gemma4_e2b.ipynb` — **main deliverable**
- [ ] 13. Cell: install libs + check the GPU
- [ ] 14. Cell: log in to Hugging Face
- [ ] 15. Cell: load the 4-bit model + tokenizer
- [ ] 16. Cell: load and format the dataset
- [ ] 17. Cell: **train A — LoRA**
- [ ] 18. Cell: **train B — KronA**
- [ ] 19. Cell: compare (params, VRAM, time, loss, samples)
- [ ] 20. Cell: save both adapters
- [ ] 21. Validate end-to-end on Colab and fix issues
- [ ] 22. Send me any error you hit *(you)*

---

## Rules

- Both runs use the **same data, seed, steps, learning rate, batch size and target modules** — only the adapter type differs.
- Cap the dataset with one easy knob (`MAX_SAMPLES`) so a run stays fast.
