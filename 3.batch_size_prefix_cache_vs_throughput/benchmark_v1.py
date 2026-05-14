"""
vLLM Throughput vs Batch Size Benchmark
----------------------------------------
Before running, start the vLLM server with --max-num-seqs matching your target batch size:

    vllm serve Qwen/Qwen3-4B-Instruct-2507 \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 4096 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 16384 \
        --dtype auto --gpu-memory-utilization 0.9

Then run:
    python benchmark_v1.py
"""

import sys
import time
import json
import threading
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
VLLM_BASE_URL   = "https://fc71qrkv8gmdqh-8000.proxy.runpod.net"
MODEL           = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT          = "Explain the theory of relativity in detail, covering special and general relativity."
MAX_TOKENS      = 256
BATCH_SIZES     = [1, 2, 4, 8, 16, 32]
DURATION_SECS   = 90     # seconds per batch size run
RAMP_PAUSE_SECS = 5      # cooldown between runs
RESULTS_DIR     = Path("bench_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
def check_server():
    print(f"\n{'='*60}")
    print(f"  Checking vLLM at {VLLM_BASE_URL} ...")
    print(f"{'='*60}")
    try:
        r = requests.get(f"{VLLM_BASE_URL}/health", timeout=10)
        if r.status_code == 200:
            print("  ✓ Server is reachable and healthy\n")
            return True
    except Exception as e:
        print(f"  ✗ Cannot reach server: {e}")
        print("  → Make sure vLLM is running on the remote machine and port 8000 is open.")
        sys.exit(1)


# ── SINGLE STREAMING REQUEST ──────────────────────────────────────────────────
def single_request(session, stop_event):
    if stop_event.is_set():
        return None

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    t_start       = time.perf_counter()
    t_first_token = None
    token_count   = 0

    try:
        with session.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=120,
        ) as resp:
            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            for raw_line in resp.iter_lines():
                if stop_event.is_set():
                    break
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        token_count += 1
                except Exception:
                    pass

    except Exception as e:
        return {"success": False, "error": str(e)}

    t_end    = time.perf_counter()
    total_ms = (t_end - t_start) * 1000

    if t_first_token is None or token_count < 2:
        return {"success": False, "error": "insufficient tokens"}

    ttft_ms = (t_first_token - t_start) * 1000
    tpot_ms = (total_ms - ttft_ms) / (token_count - 1)
    tput    = token_count / (t_end - t_start)

    return {
        "success":    True,
        "ttft_ms":    ttft_ms,
        "tpot_ms":    tpot_ms,
        "total_ms":   total_ms,
        "tokens":     token_count,
        "throughput": tput,
    }


# ── RUN ONE CONCURRENCY LEVEL ─────────────────────────────────────────────────
def run_sweep(concurrency: int) -> dict:
    print(f"\n  ▶ Concurrency = {concurrency}  |  Duration = {DURATION_SECS}s")
    print(f"  {'─'*50}")

    sessions   = [requests.Session() for _ in range(concurrency)]
    ttft_list  = []
    tpot_list  = []
    tput_list  = []
    errors     = 0
    lock       = threading.Lock()
    stop_event = threading.Event()

    def worker(session):
        nonlocal errors
        while not stop_event.is_set():
            result = single_request(session, stop_event)
            if result is None:
                break
            with lock:
                if result["success"]:
                    ttft_list.append(result["ttft_ms"])
                    tpot_list.append(result["tpot_ms"])
                    tput_list.append(result["throughput"])
                else:
                    errors += 1

    t_run_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i in range(concurrency):
            pool.submit(worker, sessions[i % len(sessions)])

        while time.time() - t_run_start < DURATION_SECS:
            elapsed = time.time() - t_run_start
            sys.stdout.write(
                f"\r    ⏱  {elapsed:>5.1f}s / {DURATION_SECS}s  |  "
                f"completed={len(ttft_list):>4}  errors={errors}"
            )
            sys.stdout.flush()
            time.sleep(1)

        stop_event.set()

    print()  # newline after progress line

    def pct(data, p):
        return round(float(np.percentile(data, p)), 2) if data else 0.0

    def mean(data):
        return round(float(np.mean(data)), 2) if data else 0.0

    result = {
        "concurrency":   concurrency,
        "total_requests": len(ttft_list),
        "errors":        errors,
        "ttft": {
            "p50": pct(ttft_list, 50),
            "p90": pct(ttft_list, 90),
            "p99": pct(ttft_list, 99),
            "mean": mean(ttft_list),
        },
        "tpot": {
            "p50": pct(tpot_list, 50),
            "p90": pct(tpot_list, 90),
            "p99": pct(tpot_list, 99),
            "mean": mean(tpot_list),
        },
        "throughput": {
            "p50":  pct(tput_list, 50),
            "p90":  pct(tput_list, 90),
            "mean": mean(tput_list),
        },
    }

    print(f"    TTFT   p50={result['ttft']['p50']}ms   p90={result['ttft']['p90']}ms   p99={result['ttft']['p99']}ms")
    print(f"    TPOT   p50={result['tpot']['p50']}ms   p90={result['tpot']['p90']}ms   p99={result['tpot']['p99']}ms")
    print(f"    Tput   mean={result['throughput']['mean']} tok/s")
    return result


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def plot_results(all_results):
    print(f"\n{'='*60}")
    print("  Generating plots ...")
    print(f"{'='*60}")

    concs      = [r["concurrency"]           for r in all_results]
    ttft_p50   = [r["ttft"]["p50"]           for r in all_results]
    ttft_p90   = [r["ttft"]["p90"]           for r in all_results]
    ttft_p99   = [r["ttft"]["p99"]           for r in all_results]
    tpot_p50   = [r["tpot"]["p50"]           for r in all_results]
    tpot_p90   = [r["tpot"]["p90"]           for r in all_results]
    tput_mean  = [r["throughput"]["mean"]    for r in all_results]
    total_reqs = [r["total_requests"]        for r in all_results]
    sys_tput   = [r["concurrency"] * r["throughput"]["mean"] for r in all_results]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"vLLM Benchmark  —  {MODEL}\n"
        f"Remote: {VLLM_BASE_URL}  |  GPU: NVIDIA RTX PRO 4500 (32GB)  |  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1. TTFT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(concs, ttft_p50, "o-",  label="p50", color="#2196F3", linewidth=2)
    ax1.plot(concs, ttft_p90, "s--", label="p90", color="#FF9800", linewidth=2)
    ax1.plot(concs, ttft_p99, "^:",  label="p99", color="#F44336", linewidth=2)
    ax1.set_title("TTFT — Time to First Token", fontweight="bold")
    ax1.set_xlabel("Concurrent Users (batch size)")
    ax1.set_ylabel("Latency (ms)")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xticks(concs)

    # 2. TPOT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(concs, tpot_p50, "o-",  label="p50", color="#4CAF50", linewidth=2)
    ax2.plot(concs, tpot_p90, "s--", label="p90", color="#FF9800", linewidth=2)
    ax2.set_title("TPOT — Time Per Output Token", fontweight="bold")
    ax2.set_xlabel("Concurrent Users (batch size)")
    ax2.set_ylabel("ms / token")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xticks(concs)

    # 3. Per-request throughput
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar([str(c) for c in concs], tput_mean, color="#9C27B0", alpha=0.85)
    ax3.set_title("Throughput per Request", fontweight="bold")
    ax3.set_xlabel("Concurrent Users (batch size)")
    ax3.set_ylabel("tok/s")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. System-wide throughput
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(concs, sys_tput, "D-", color="#009688", linewidth=2, markersize=9)
    for x, y in zip(concs, sys_tput):
        ax4.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax4.set_title("System Throughput (users × tok/s)", fontweight="bold")
    ax4.set_xlabel("Concurrent Users (batch size)")
    ax4.set_ylabel("Total tok/s")
    ax4.grid(True, alpha=0.3); ax4.set_xticks(concs)

    # 5. Completed requests per run
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar([str(c) for c in concs], total_reqs, color="#607D8B", alpha=0.85)
    for i, v in enumerate(total_reqs):
        ax5.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    ax5.set_title(f"Completed Requests / {DURATION_SECS}s", fontweight="bold")
    ax5.set_xlabel("Concurrent Users (batch size)")
    ax5.set_ylabel("# Requests")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. TTFT p50 vs TPOT p50 scatter
    ax6 = fig.add_subplot(gs[1, 2])
    sc = ax6.scatter(ttft_p50, tpot_p50, c=concs, cmap="plasma", s=140, zorder=5)
    for i, c in enumerate(concs):
        ax6.annotate(f"u={c}", (ttft_p50[i], tpot_p50[i]),
                     textcoords="offset points", xytext=(7, 4), fontsize=8)
    plt.colorbar(sc, ax=ax6, label="Concurrency")
    ax6.set_title("TTFT p50 vs TPOT p50", fontweight="bold")
    ax6.set_xlabel("TTFT p50 (ms)")
    ax6.set_ylabel("TPOT p50 (ms/tok)")
    ax6.grid(True, alpha=0.3)

    out_path = RESULTS_DIR / "benchmark_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Plot saved → {out_path}")
    plt.close()


# ── PRINT FINAL TABLE ─────────────────────────────────────────────────────────
def print_summary(all_results):
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY TABLE")
    print(f"{'='*60}")
    header = f"{'Users':>6} | {'TTFT p50':>9} | {'TTFT p99':>9} | {'TPOT p50':>9} | {'Tput mean':>10} | {'Reqs':>6}"
    print(header)
    print("─" * len(header))
    for r in all_results:
        print(
            f"{r['concurrency']:>6} | "
            f"{r['ttft']['p50']:>8.1f}ms | "
            f"{r['ttft']['p99']:>8.1f}ms | "
            f"{r['tpot']['p50']:>8.2f}ms | "
            f"{r['throughput']['mean']:>9.1f}/s | "
            f"{r['total_requests']:>6}"
        )
    print(f"{'='*60}\n")


# ── SAVE JSON ─────────────────────────────────────────────────────────────────
def save_json(all_results):
    import json as _json
    out = {
        "model": MODEL,
        "remote": VLLM_BASE_URL,
        "prompt_tokens": len(PROMPT.split()),
        "max_output_tokens": MAX_TOKENS,
        "duration_per_sweep_secs": DURATION_SECS,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    path = RESULTS_DIR / "benchmark_results.json"
    path.write_text(_json.dumps(out, indent=2))
    print(f"  ✓ JSON saved  → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  LLM BENCHMARK SUITE")
    print(f"  Model  : {MODEL}")
    print(f"  Remote : {VLLM_BASE_URL}")
    print(f"  Sweeps : {BATCH_SIZES}")
    print(f"  Duration/sweep: {DURATION_SECS}s")
    print(f"{'='*60}")

    check_server()

    all_results = []
    for i, batch in enumerate(BATCH_SIZES):
        result = run_sweep(batch)
        all_results.append(result)
        if i < len(BATCH_SIZES) - 1:
            print(f"\n  ⏸  Cooling down {RAMP_PAUSE_SECS}s before next run...")
            time.sleep(RAMP_PAUSE_SECS)

    print_summary(all_results)
    save_json(all_results)
    plot_results(all_results)

    print("\n  ✅ Benchmark complete!")
    print(f"  Results in: ./{RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()