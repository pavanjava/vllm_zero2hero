# SGLang vs vLLM — Inference Server Benchmarking Experiment

## Overview

This experiment benchmarks two popular open-source LLM inference servers — **SGLang** and **vLLM** — on identical hardware using the same model and a demanding set of prompts. The goal is to provide a realistic, data-driven answer to the question: *which inference server should you choose?*

---

## Hardware & Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA L40S (46 GiB VRAM) |
| CPU | AMD EPYC 9354 |
| RAM | 87–175 GiB |
| CUDA Version | 12.4 / 12.8 |
| Driver Version | 550.127.08 / 570.124.06 |
| OS | Ubuntu 22.04 LTS |
| Deployment | RunPod cloud instance |

---

## Model

**Qwen/Qwen3-4B-Instruct-2507**
- 4B parameter instruction-tuned model with built-in reasoning/thinking capability
- Served via OpenAI-compatible `/v1/chat/completions` endpoint

---

## Prompt Design

Rather than using simple or toy prompts, this experiment uses **100 Irodov-style hard physics and mathematics problems** drawn from the following domains:

- Classical Mechanics
- Thermodynamics
- Electrostatics & Magnetism
- Optics & Wave Physics
- Modern Physics & Quantum Mechanics
- Advanced Mechanics (Lagrangian, Gyroscopes, Kepler)
- Electrodynamics & Maxwell's Equations
- Statistical Mechanics

This ensures the benchmark reflects real inference workload — long context, complex reasoning chains, and multi-step mathematical derivations — not trivial question-answering.

---

## Load Test Configuration

| Parameter | Value |
|---|---|
| Tool | Locust 2.43.4 |
| Concurrent Users | 100 |
| Spawn Rate | 5 users/second |
| Duration | ~5–10 minutes |
| max_tokens | 512 |
| Temperature | 0.7 |
| Timeout per request | 120 seconds |

---

## Results

### SGLang

| Metric | Value |
|---|---|
| Total Requests | 1,968 |
| Failures | 0 (0.00%) |
| RPS (steady-state) | ~6.55 |
| Median Latency (p50) | 13,000 ms |
| p95 Latency | 13,000 ms |
| p99 Latency | 14,000 ms |
| Min Latency | 5,944 ms |
| Max Latency | 14,395 ms |
| GPU Utilisation | 97% |
| VRAM Usage | 43 GiB / 45 GiB (94%) |
| GPU Temperature | 62–72°C |
| GPU Power | 291–326W / 350W |

### vLLM

| Metric | Value |
|---|---|
| Total Requests | 1,598 |
| Failures | 0 (0.00%) |
| RPS (steady-state) | ~7.25 |
| Median Latency (p50) | 11,000 ms |
| p95 Latency | 12,000 ms |
| p99 Latency | 12,000 ms |
| Min Latency | 6,176 ms |
| Max Latency | 12,092 ms |
| GPU Utilisation | 100% |
| VRAM Usage | 37 GiB / 45 GiB (81%) |
| GPU Temperature | 57–64°C |
| GPU Power | 324W / 350W |

---

## Head-to-Head Comparison

| Metric | SGLang | vLLM | Winner |
|---|---|---|---|
| Throughput (RPS) | 6.55 | **7.25** | ✅ vLLM (+10.7%) |
| Median Latency | 13,000 ms | **11,000 ms** | ✅ vLLM (-15.4%) |
| p95 Latency | 13,000 ms | **12,000 ms** | ✅ vLLM (-7.7%) |
| VRAM Usage | 94% | **81%** | ✅ vLLM (-13%) |
| GPU Temperature | 72°C | **64°C** | ✅ vLLM (-8°C) |
| Failure Rate | 0% | 0% | 🟰 Tie |

**vLLM wins across every measurable metric on this hardware and workload.**

---

## Key Observations

1. **vLLM delivers ~10% higher throughput** (7.25 vs 6.55 RPS) under identical load.
2. **vLLM is ~2 seconds faster at median latency** (11s vs 13s), which is significant for interactive use cases.
3. **vLLM uses 13% less VRAM** (37 GiB vs 43 GiB), which means more headroom for larger models or longer context windows.
4. **vLLM runs cooler** (64°C vs 72°C), indicating more efficient GPU utilisation.
5. **Both servers achieved 0% failure rate**, confirming production-grade reliability for both.
6. **SGLang has a more complex setup** — requires `libnuma` system library and `sgl_kernel` compatibility with the specific GPU compute capability (SM89 for L40S).

---

## Repository Structure

```
.
├── sgl_inferencing/
│   ├── __init__.py
│   └── locustfile.py        # SGLang load test
├── vllm_inferencing/
│   ├── __init__.py
│   └── locustfile.py        # vLLM load test
└── .gitignore
```

---

## Installation Notes

### SGLang
```bash
# System dependency (critical — must be installed first)
apt-get update && apt-get install -y libnuma1 libnuma-dev

# Create venv and install
python3 -m venv /opt/sglang-env
source /opt/sglang-env/bin/activate
pip install --upgrade pip
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/

# Serve
python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --dtype auto --host 0.0.0.0 --port 8000
```

### vLLM
```bash
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm

# Serve
vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype auto \
  --host 0.0.0.0 --port 8000
```

### Load Test
```bash
pip install locust
locust -f locustfile.py --headless --users 100 --spawn-rate 5 \
  --run-time 10m --html report.html
```

---

## Conclusion

For production inference on a single NVIDIA L40S with a 4B-class reasoning model, **vLLM is the recommended choice** — it delivers better throughput, lower latency, and more efficient VRAM utilisation out of the box. SGLang remains a strong option for workloads involving complex chained prompts or structured generation, but requires more careful environment setup.

---

*Experiment conducted by Pavan Kumar Mantha | April 2026*