# VRAM Profiler

GPU memory profiling tool for LLM inference.

---

## Install

```bash
pip install torch transformers
```

---

## Run

```bash
python vram_profiler.py
```

---

## What It Does

Profiles GPU memory usage across 7 stages:

1. **CUDA Baseline** - Runtime overhead
2. **Model Load** - Weight memory
3. **Forward Pass** - Activation memory at different sequence lengths
4. **Per-Layer** - Memory per transformer layer
5. **KV Cache** - Cache growth during generation
6. **Fragmentation** - Memory efficiency analysis
7. **AI Analysis** - Auto-generated insights

---

## Output

Creates `vram_profile_report.md` with:
- Memory metrics for each stage
- Allocated vs reserved vs peak memory
- Fragmentation analysis
- AI-generated recommendations

---

## Customize

Change model:
```python
model_id = "meta-llama/Llama-3-8B-Instruct"
```

Change precision:
```python
torch_dtype = torch.float16  # or torch.bfloat16
```

---

## Use For

- Estimating GPU requirements
- Finding memory bottlenecks
- Debugging OOM errors
- Planning batch sizes
- Comparing models