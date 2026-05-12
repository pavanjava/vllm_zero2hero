# vLLM Throughput vs Batch Size Benchmark

Demonstrates how throughput, TTFT, and TPOT vary with batch size using `vllm serve` and concurrent streaming HTTP requests.

---

## Setup

```bash
pip install vllm httpx
```

---

## How It Works

- `--max-num-seqs` on the server controls how many requests vLLM processes simultaneously
- The client sends exactly `N` concurrent **streaming** requests to match the server's batch capacity
- Streaming is required to capture **TTFT** — the timestamp of the first token chunk is recorded as it arrives
- **Rule:** `client batch_size == server --max-num-seqs` for a clean experiment

---

## Metrics Explained

| Metric | Full Name | Formula | What It Measures |
|--------|-----------|---------|-----------------|
| **TTFT** | Time to First Token | `t_first_token - t_request_sent` | Prefill phase latency — how long before the user sees anything |
| **TPOT** | Time Per Output Token | `(total_latency - TTFT) / (output_tokens - 1)` | Decode phase speed — how fast tokens stream after the first |
| **Agg. TPS** | Aggregate Tokens/sec | `total_tokens / wall_clock_time` | Overall GPU throughput across all concurrent requests |

> **TTFT** is dominated by prompt length and prefill compute.  
> **TPOT** is dominated by model size, batch size, and KV cache pressure.

---

## Step 1: Start the Server

Pick your target batch size and start the server accordingly:

```bash
# Example: batch_size = 8
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
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

## Step 2: Run the Benchmark V1

```bash
python benchmark_v1.py
```

---

## Sample Output
```commandline
============================================================
  FINAL SUMMARY TABLE
============================================================
 Users |  TTFT p50 |  TTFT p99 |  TPOT p50 |  Tput mean |   Reqs
────────────────────────────────────────────────────────────────
     1 |   1230.1ms |   1496.9ms |    11.18ms |      65.9/s |     24
     2 |    628.1ms |   9809.6ms |    11.56ms |      66.0/s |     44
     4 |    559.9ms |   1498.0ms |    11.73ms |      67.0/s |     96
     8 |    558.5ms |   2002.7ms |    11.95ms |      65.0/s |    177
    16 |    571.4ms |   2198.1ms |    12.20ms |      65.7/s |    377
    32 |   1143.3ms |   4204.1ms |    12.81ms |      56.5/s |    633
============================================================
```

## Step 3: Run the Benchmark V2

```bash
python benchmark_v2.py
```

---

## Sample Output
```commandline
============================================================
 Users |  TTFT p50 |  TTFT p99 |  TPOT p50 |  Tput mean |   Reqs
────────────────────────────────────────────────────────────────
     1 |   1246.1ms |  34335.0ms |    11.35ms |      59.5/s |     13
     2 |   1165.1ms |   1373.3ms |    11.66ms |      66.8/s |     47
     4 |    525.0ms |   1342.4ms |    11.81ms |      67.3/s |     96
     8 |    496.3ms |   2494.0ms |    11.83ms |      68.8/s |    190
    16 |    523.1ms |   2949.9ms |    12.09ms |      65.0/s |    369
    32 |    677.8ms |   4842.4ms |    12.66ms |      59.6/s |    664
============================================================
```