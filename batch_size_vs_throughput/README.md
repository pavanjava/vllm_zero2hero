# vLLM Throughput vs Batch Size Benchmark

Demonstrates how throughput (tokens/sec) varies with batch size using `vllm serve` and concurrent HTTP requests.

---

## Setup

```bash
pip install vllm httpx
```

---

## How It Works

- `--max-num-seqs` on the server controls how many requests vLLM processes simultaneously
- The client sends exactly `N` concurrent requests to match the server's batch capacity
- **Rule:** `client batch_size == server --max-num-seqs` for a clean experiment

---

## Step 1: Start the Server

Pick your target batch size and start the server accordingly:

```bash
# batch_size = 8
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.9
```

| Batch Size | `--max-num-seqs` | `--max-num-batched-tokens` |
|------------|-----------------|---------------------------|
| 1          | 1               | 512                        |
| 4          | 4               | 2048                       |
| 8          | 8               | 4096                       |
| 16         | 16              | 8192                       |
| 32         | 32              | 16384                      |

> Scale `--max-num-batched-tokens` = `--max-num-seqs × avg_tokens_per_request`

---

## Step 2: Run the Benchmark

```bash
python benchmark.py
```

---

## Sample Output

```
Batch Size |  Total Time (s) |    Agg. TPS | Avg Req TPS
------------------------------------------------------------
         1 |           1.243 |      103.00 |      103.00
         4 |           1.512 |      338.62 |       84.65
         8 |           1.891 |      541.51 |       67.69
        16 |           2.340 |      874.36 |       54.65
        32 |           3.102 |     1317.22 |       41.16
```

---

## Key Insight

| Metric | Trend with ↑ Batch Size |
|--------|------------------------|
| Aggregate TPS | ↑ Increases (GPU better utilized) |
| Per-request TPS | ↓ Decreases (each request waits longer) |
| Total latency | ↑ Increases slightly |

Throughput scales with batch size until GPU memory or compute saturates.