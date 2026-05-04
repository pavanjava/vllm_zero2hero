"""
vLLM Throughput vs Batch Size Benchmark
----------------------------------------
Before running, start the vLLM server with --max-num-seqs matching your target batch size:

    vllm serve Qwen/Qwen2.5-1.5B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --max-num-seqs <BATCH_SIZE> \
        --max-num-batched-tokens <BATCH_SIZE * 512> \
        --dtype auto --gpu-memory-utilization 0.9

Then run:
    python benchmark.py
"""

import asyncio
import time
import httpx

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "http://localhost:8000/v1/completions"
MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT     = "Explain the theory of relativity in simple terms."
MAX_TOKENS = 128

# Must match the server's --max-num-seqs for each run
BATCH_SIZES = [1, 4, 8, 16, 32]
# ─────────────────────────────────────────────────────────────────────────────


async def send_request(client: httpx.AsyncClient) -> dict:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    response = await client.post(BASE_URL, json=payload, timeout=120)
    latency = time.perf_counter() - t0

    response.raise_for_status()
    tokens_out = response.json()["usage"]["completion_tokens"]
    return {"latency": latency, "tokens": tokens_out}


async def run_batch(batch_size: int) -> dict:
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[send_request(client) for _ in range(batch_size)])
        total_elapsed = time.perf_counter() - t0

    total_tokens = sum(r["tokens"] for r in results)
    avg_latency  = sum(r["latency"] for r in results) / len(results)

    return {
        "batch_size"          : batch_size,
        "total_time_sec"      : round(total_elapsed, 3),
        "aggregate_tps"       : round(total_tokens / total_elapsed, 2),
        "avg_latency_sec"     : round(avg_latency, 3),
    }


async def main():
    print(f"\n{'Batch Size':>12} | {'Total Time (s)':>15} | {'Agg. TPS':>10} | {'Avg Latency (s)':>16}")
    print("-" * 62)

    for bs in BATCH_SIZES:
        try:
            r = await run_batch(bs)
            print(
                f"{r['batch_size']:>12} | "
                f"{r['total_time_sec']:>15} | "
                f"{r['aggregate_tps']:>10} | "
                f"{r['avg_latency_sec']:>16}"
            )
        except Exception as e:
            print(f"{bs:>12} | ERROR: {e}")

    print("\nNote: Restart server with --max-num-seqs=<batch_size> between runs for accurate results.")


if __name__ == "__main__":
    asyncio.run(main())