# vram_profiler.py
# Hands-on VRAM profiling on A40, A100, L40, L40s
# Requirements: pip install torch transformers
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

# ─── Utility ──────────────────────────────────────────────────────────────────
model_id="Qwen/Qwen3-4B-Instruct-2507"
print(f"  Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(f"  Loading model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
model.eval()

def mb(bytes_val: int) -> float:
    return bytes_val / (1024 ** 2)

def snapshot(label: str) -> Dict[str, float]:
    torch.cuda.synchronize()
    stats = {
        "label":     label,
        "allocated": mb(torch.cuda.memory_allocated()),
        "reserved":  mb(torch.cuda.memory_reserved()),
        "peak":      mb(torch.cuda.max_memory_allocated()),
    }
    print(f"[{label:30s}] alloc={stats['allocated']:7.1f} MB | "
          f"reserved={stats['reserved']:7.1f} MB | "
          f"peak={stats['peak']:7.1f} MB")
    return stats

def reset_peak():
    torch.cuda.reset_peak_memory_stats()

# ─── Stage 1: Baseline (CUDA runtime cost) ───────────────────────────────────

def stage1_baseline():
    print("\n" + "="*70)
    print("STAGE 1 — Baseline CUDA runtime overhead")
    print("="*70)
    reset_peak()
    # Just initializing CUDA context costs ~300–600 MB on A100
    _ = torch.zeros(1).cuda()
    torch.cuda.synchronize()
    s = snapshot("CUDA context init")
    print(f"  → Baseline overhead: {s['allocated']:.1f} MB allocated")
    return s

# ─── Stage 2: Model load VRAM ─────────────────────────────────────────────────

def stage2_model_load(model_id: str = "Qwen/Qwen3-4B"):
    print("\n" + "="*70)
    print(f"STAGE 2 — Model load: {model_id}")
    print("="*70)

    before = mb(torch.cuda.memory_allocated())
    reset_peak()

    s = snapshot("After model load")
    weight_vram = s["allocated"] - before
    print(f"  → Model weights consumed: {weight_vram:.1f} MB")

    param_count = sum(p.numel() for p in model.parameters())
    theoretical = param_count * 2 / (1024**2)  # bfloat16 = 2 bytes
    print(f"  → Params: {param_count/1e9:.2f}B | Theoretical BF16: {theoretical:.1f} MB")
    print(f"  → Overhead vs theoretical: {weight_vram - theoretical:.1f} MB")

    return s

# ─── Stage 3: Forward pass peak (activation spike) ───────────────────────────

def stage3_forward_pass(seq_len: int = 512):
    print("\n" + "="*70)
    print(f"STAGE 3 — Forward pass (seq_len={seq_len})")
    print("="*70)

    before_alloc = mb(torch.cuda.memory_allocated())
    reset_peak()

    text = "The quick brown fox " * (seq_len // 5)
    inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to("cuda:0")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    torch.cuda.synchronize()
    s = snapshot(f"Forward pass seq={seq_len}")
    activation_spike = s["peak"] - before_alloc
    print(f"  → Activation spike above weights: {activation_spike:.1f} MB")

    del outputs, inputs
    torch.cuda.empty_cache()
    return s, activation_spike

# ─── Stage 4: Per-layer VRAM via forward hooks ───────────────────────────────

def stage4_per_layer(seq_len: int = 256):
    print("\n" + "="*70)
    print(f"STAGE 4 — Per-layer VRAM breakdown (first 8 transformer blocks)")
    print("="*70)

    layer_stats: List[Dict] = []
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            torch.cuda.synchronize()
            layer_stats.append({
                "layer": layer_idx,
                "allocated": mb(torch.cuda.memory_allocated()),
                "peak":      mb(torch.cuda.max_memory_allocated()),
            })
        return hook

    # Register hooks on first 8 transformer layers
    # Qwen3 uses model.model.layers; adjust for other architectures
    layers = model.model.layers
    for i, layer in enumerate(layers[:8]):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    text = "Explain attention mechanisms in transformers. " * (seq_len // 10)
    inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to("cuda:0")

    reset_peak()
    with torch.no_grad():
        model(**inputs, use_cache=False)
    torch.cuda.synchronize()

    for h in hooks:
        h.remove()

    print(f"\n  {'Layer':>6} | {'Allocated MB':>14} | {'Peak MB':>10} | {'Delta MB':>10}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}")
    prev_alloc = layer_stats[0]["allocated"] if layer_stats else 0
    for s in layer_stats:
        delta = s["allocated"] - prev_alloc
        print(f"  {s['layer']:>6} | {s['allocated']:>14.1f} | {s['peak']:>10.1f} | {delta:>+10.1f}")
        prev_alloc = s["allocated"]

    del inputs
    torch.cuda.empty_cache()
    return layer_stats

# ─── Stage 5: KV cache growth simulation ─────────────────────────────────────

def stage5_kv_cache():
    print("\n" + "="*70)
    print("STAGE 5 — KV cache growth (simulated decode)")
    print("="*70)

    base_text = "The transformer architecture has revolutionized natural language processing."
    inputs = tokenizer(base_text, return_tensors="pt").to("cuda")

    kv_measurements = []
    past_key_values = None

    print(f"\n  {'Step':>6} | {'Total Alloc MB':>16} | {'KV delta MB':>12}")
    print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*12}")

    prev = mb(torch.cuda.memory_allocated())

    with torch.no_grad():
        for step in range(10):  # simulate 10 decode steps
            if past_key_values is None:
                out = model(**inputs, use_cache=True)
            else:
                # Feed only last token
                last_token = torch.tensor([[generated_token]]).to("cuda:0")
                out = model(input_ids=last_token, past_key_values=past_key_values,
                            use_cache=True)

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            generated_token = logits.argmax(-1).item()

            torch.cuda.synchronize()
            curr = mb(torch.cuda.memory_allocated())
            delta = curr - prev
            kv_measurements.append({"step": step, "allocated": curr, "delta": delta})
            print(f"  {step:>6} | {curr:>16.1f} | {delta:>+12.2f}")
            prev = curr

    del past_key_values
    torch.cuda.empty_cache()
    return kv_measurements

# ─── Stage 6: Fragmentation analysis ─────────────────────────────────────────

def stage6_fragmentation():
    print("\n" + "="*70)
    print("STAGE 6 — Fragmentation: allocated vs reserved vs peak")
    print("="*70)

    allocated = mb(torch.cuda.memory_allocated())
    reserved  = mb(torch.cuda.memory_reserved())
    peak      = mb(torch.cuda.max_memory_allocated())
    total     = mb(torch.cuda.get_device_properties(0).total_memory)

    fragmented   = reserved - allocated
    headroom     = total - reserved
    utilization  = (allocated / total) * 100

    print(f"\n  Total GPU VRAM  : {total:>8.1f} MB")
    print(f"  Allocated       : {allocated:>8.1f} MB  ← tensors actually in use")
    print(f"  Reserved (cache): {reserved:>8.1f} MB  ← held by PyTorch allocator")
    print(f"  Fragmented gap  : {fragmented:>8.1f} MB  ← reserved but not allocated")
    print(f"  Headroom free   : {headroom:>8.1f} MB  ← truly available")
    print(f"  Peak ever seen  : {peak:>8.1f} MB")
    print(f"  Utilization     : {utilization:>7.1f}%")

    if fragmented > 500:
        print(f"\n  ⚠  High fragmentation ({fragmented:.0f} MB). "
              f"Consider torch.cuda.empty_cache() between workloads.")
    else:
        print(f"\n  ✓  Fragmentation within normal range.")

# ─── Stage7: Analyse the report ─────────────────────────────────────────────────────────────────────

def analyse_report(md_file_path: str) -> str:

    with open(md_file_path, "r") as f:
        report_content = f.read()

    messages = [
        {
            "role": "system",
            "content": "You are an expert AI/ML infrastructure engineer. Analyse the given VRAM profiling report and provide insights on memory efficiency, bottlenecks, and recommendations."
        },
        {
            "role": "user",
            "content": f"Analyse this VRAM profiling report and give detailed insights:\n\n{report_content}"
        }
    ]

    # Apply Qwen3 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # set True if you want chain-of-thought
    )

    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens, not the prompt
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Tee stdout to both terminal and markdown file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open("vram_profile_report.md", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    assert torch.cuda.is_available(), "No CUDA device found!"
    device = torch.cuda.get_device_name(0)
    total  = mb(torch.cuda.get_device_properties(0).total_memory)
    print(f"\n{'='*70}")
    print(f"  Device : {device}")
    print(f"  VRAM   : {total:.0f} MB ({total/1024:.1f} GB)")
    print(f"{'='*70}")

    # Run all stages
    stage1_baseline()
    stage3_forward_pass(seq_len=256)
    _ = stage2_model_load("Qwen/Qwen3-4B-Instruct-2507")
    stage3_forward_pass(seq_len=1024)   # compare seq lengths
    stage4_per_layer()
    stage5_kv_cache()
    stage6_fragmentation()

    result = analyse_report("vram_profile_report.md")
    print(result)