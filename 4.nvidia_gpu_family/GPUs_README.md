# NVIDIA GPUs  — Grouped by Architecture

> Source: [/Runpod]· Pricing is marketplace-driven and fluctuates in real time.  
> **76+ GPU types** across all generations — from Pascal to Blackwell.

---

## Architecture Generations (Newest → Oldest)

| Generation | Code Name | Year | Key Feature |
|---|---|---|---|
| Blackwell | B-series / RTX 50xx / RTX PRO | 2024–2025 | FP4 Tensor Cores, 5th Gen, up to 192GB HBM3e |
| Hopper | H-series | 2022–2023 | Transformer Engine, FP8, up to 141GB HBM3e |
| Ada Lovelace | L-series / RTX 40xx / RTX 4000–6000 | 2022–2023 | 4th Gen Tensor Cores, DLSS 3, up to 48GB GDDR6 |
| Ampere | A-series / RTX 30xx / RTX A-series | 2020–2022 | 3rd Gen Tensor Cores, up to 80GB HBM2e |
| Turing | RTX 20xx / Quadro RTX / T4 | 2018–2019 | 1st Gen RT Cores, 1st Gen Tensor Cores |
| Volta | V100 / Titan V | 2017–2018 | First Tensor Cores (1st Gen) |
| Pascal | GTX 10xx / Tesla P-series / Titan Xp | 2016–2017 | CUDA 6.x, no Tensor Cores |

---

## 🟢 Blackwell (2024–2025)

Named after statistician **David Blackwell**.  
Features: 5th Gen Tensor Cores, FP4 support, GDDR7/HBM3e memory, highest throughput per watt.

### Data Center

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| B200 | 192 GB | HBM3e | ~2,200 | Contact Sales |
| B200 SXM | 192 GB | HBM3e | ~2,200 | Contact Sales |

### Workstation / Professional

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell | 48 GB | GDDR6 ECC | ~418 | ~$0.50+ |
| RTX PRO 5000 Blackwell | 32 GB | GDDR6 | — | — |
| RTX PRO 4500 Blackwell | 32 GB | GDDR6 | — | ~$0.17+ |
| RTX PRO 4000 Blackwell | 24 GB | GDDR6 | — | ~$0.17+ |

### Consumer GeForce RTX 50xx

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX 5090 | 32 GB | GDDR7 | 104.8 | ~$0.41 |
| RTX 5080 | 16 GB | GDDR7 | ~80 | ~$0.12 |
| RTX 5070 Ti | 16 GB | GDDR7 | ~60 | ~$0.06 |
| RTX 5070 | 12 GB | GDDR7 | ~50 | ~$0.06 |
| RTX 5060 Ti | 16 GB | GDDR6 | ~40 | ~$0.04 |
| RTX 5060 | 8 GB | GDDR7 | ~30 | ~$0.09 |

---

## 🔵 Hopper (2022–2023)

Named after computer scientist and US Navy Admiral **Grace Hopper**.  
Features: Transformer Engine, FP8 precision, NVLink 4.0, up to 141GB HBM3e, MIG support.

### Data Center

| GPU | VRAM | Memory Type | Memory BW | FP8 TFLOPS |  Price/hr |
|---|---|---|---|---|---|
| H200 SXM | 141 GB | HBM3e | 4.8 TB/s | ~3,958 | ~$3.48+ |
| H200 NVL | 141 GB | HBM3e | 4.8 TB/s | ~3,958 | ~$3.48+ |
| H100 SXM | 80 GB | HBM3 | 3.35 TB/s | ~3,958 | ~$1.87 |
| H100 PCIe | 80 GB | HBM2e | 2.0 TB/s | ~3,026 | ~$1.73 |
| H100 NVL | 94 GB | HBM3 | 3.9 TB/s | ~3,958 | ~$1.52 |

> **H100 SXM** = highest bandwidth, best for training.  
> **H100 NVL** = dual-GPU board with 188 GB combined VRAM, great for large LLMs.  
> **H200** = same compute as H100 but 141 GB HBM3e memory — significantly faster for memory-bound workloads.

---

## 🟡 Ada Lovelace (2022–2023)

Named after English mathematician **Ada Lovelace**.  
Features: 4th Gen Tensor Cores, DLSS 3 Frame Generation, AV1 encode, up to 48 GB GDDR6.

### Data Center / Inference Optimized

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| L40S | 48 GB | GDDR6 ECC | ~362 | ~$0.53 |
| L40 | 48 GB | GDDR6 ECC | ~181 | ~$0.40+ |
| L4 | 24 GB | GDDR6 ECC | ~121 | ~$0.25+ |

### Workstation / Professional

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX 6000 Ada | 48 GB | GDDR6 ECC | ~362 | ~$0.80+ |
| RTX 5000 Ada | 32 GB | GDDR6 ECC | ~221 | ~$0.40+ |
| RTX 4500 Ada | 24 GB | GDDR6 ECC | ~165 | ~$0.30+ |
| RTX 4000 Ada | 20 GB | GDDR6 ECC | ~121 | ~$0.14 |
| RTX 4000 SFF Ada | 20 GB | GDDR6 ECC | ~121 | — |
| RTX 2000 Ada | 16 GB | GDDR6 ECC | ~57 | ~$0.10 |

### Consumer GeForce RTX 40xx

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX 4090 | 24 GB | GDDR6X | ~165 | ~$0.31 |
| RTX 4080 Super | 16 GB | GDDR6X | ~121 | ~$0.17 |
| RTX 4080 | 16 GB | GDDR6X | ~113 | ~$0.16 |
| RTX 4070 Ti Super | 16 GB | GDDR6X | ~92 | ~$0.11 |
| RTX 4070 Ti | 12 GB | GDDR6X | ~82 | ~$0.08 |
| RTX 4070 Super | 12 GB | GDDR6X | ~71 | ~$0.09 |
| RTX 4070 | 12 GB | GDDR6 | ~60 | ~$0.07 |
| RTX 4070 Laptop | 8 GB | GDDR6 | ~36 | — |
| RTX 4060 Ti | 8–16 GB | GDDR6 | ~45 | ~$0.05 |
| RTX 4060 | 8 GB | GDDR6 | ~31 | ~$0.04 |

---

## 🟠 Ampere (2020–2022)

Named after French mathematician and physicist **André-Marie Ampère**.  
Features: 3rd Gen Tensor Cores, BF16 support, A100 with HBM2e, MIG support on A100/A30.

### Data Center

| GPU | VRAM | Memory Type | Memory BW | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|---|
| A100 SXM4 | 80 GB | HBM2e | 2.0 TB/s | ~312 | ~$0.83 |
| A100 PCIe | 80 GB | HBM2e | 1.935 TB/s | ~312 | ~$0.65 |
| A100 40GB PCIe | 40 GB | HBM2 | 1.555 TB/s | ~312 | ~$0.65 |
| A40 | 48 GB | GDDR6 ECC | 696 GB/s | ~150 | ~$0.40+ |
| A30 | 24 GB | HBM2 | 933 GB/s | ~165 | ~$0.30+ |
| A10 | 24 GB | GDDR6 | 600 GB/s | ~125 | ~$0.20+ |
| A16 | 64 GB (4×16) | GDDR6 | 4×200 GB/s | ~4×63 | — |
| A2 | 16 GB | GDDR6 | 200 GB/s | ~25 | — |

### Workstation / Professional (RTX A-Series)

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX A6000 | 48 GB | GDDR6 ECC | ~155 | ~$0.40+ |
| RTX A5500 | 24 GB | GDDR6 ECC | ~102 | — |
| RTX A5000 | 24 GB | GDDR6 ECC | ~89 | ~$0.20+ |
| RTX A4500 | 20 GB | GDDR6 ECC | ~70 | — |
| RTX A4000 | 16 GB | GDDR6 ECC | ~77 | ~$0.05 |
| RTX A2000 | 12 GB | GDDR6 ECC | ~32 | ~$0.03 |

### Consumer GeForce RTX 30xx

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX 3090 Ti | 24 GB | GDDR6X | ~80 | ~$0.19 |
| RTX 3090 | 24 GB | GDDR6X | ~71 | ~$0.13 |
| RTX 3080 Ti | 12 GB | GDDR6X | ~65 | ~$0.08 |
| RTX 3080 (12GB) | 12 GB | GDDR6X | ~60 | ~$0.05+ |
| RTX 3080 | 10 GB | GDDR6X | ~60 | ~$0.03 |
| RTX 3070 Ti | 8 GB | GDDR6X | ~43 | ~$0.05 |
| RTX 3070 | 8 GB | GDDR6 | ~40 | ~$0.05 |
| RTX 3060 Ti | 8 GB | GDDR6 | ~33 | ~$0.03 |
| RTX 3060 | 12 GB | GDDR6 | ~26 | ~$0.03 |
| RTX 3060 Laptop | 6 GB | GDDR6 | ~20 | ~$0.04 |
| RTX 3050 | 8 GB | GDDR6 | ~18 | ~$0.05 |

---

## ⚪ Turing (2018–2019)

Named after mathematician and computer scientist **Alan Turing**.  
Features: 1st Gen RT Cores, 2nd Gen Tensor Cores, first real-time ray tracing consumer GPUs.

### Data Center

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| T4 | 16 GB | GDDR6 | ~65 | ~$0.15 |

### Workstation / Professional (Quadro RTX)

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| Quadro RTX 6000 | 24 GB | GDDR6 ECC | ~67 | ~$0.15+ |
| Quadro RTX 5000 | 16 GB | GDDR6 ECC | ~57 | — |
| Quadro RTX 4000 | 8 GB | GDDR6 ECC | ~29 | ~$0.11 |

### Consumer GeForce RTX 20xx

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| RTX 2080 Ti | 11 GB | GDDR6 | ~54 | ~$0.04 |
| RTX 2080 Super | 8 GB | GDDR6 | ~42 | — |
| RTX 2080 | 8 GB | GDDR6 | ~37 | — |
| RTX 2070 Super | 8 GB | GDDR6 | ~36 | ~$0.06 |
| RTX 2070 | 8 GB | GDDR6 | ~29 | ~$0.03 |
| RTX 2060 Super | 8 GB | GDDR6 | ~28 | ~$0.03 |
| RTX 2060 | 6 GB | GDDR6 | ~24 | ~$0.04 |

---

## 🔴 Volta (2017–2018)

Named after Italian physicist **Alessandro Volta**.  
Features: First-ever Tensor Cores (1st Gen), NVLink 2.0, first architecture purpose-built for deep learning.

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| Tesla V100 SXM2 | 32 GB | HBM2 | ~125 | ~$0.20+ |
| Tesla V100 PCIe | 16–32 GB | HBM2 | ~112 | ~$0.02 |
| Titan V | 12 GB | HBM2 | ~110 | ~$0.07 |

---

## 🟤 Pascal (2016–2017)

Named after French mathematician **Blaise Pascal**.  
No Tensor Cores. Still available on  for very low-cost, low-VRAM workloads.

### Data Center / Tesla

| GPU | VRAM | Memory Type | FP16 TFLOPS |  Price/hr |
|---|---|---|---|---|
| Tesla P100 | 16 GB | HBM2 | ~21 | ~$0.14 |
| Tesla P40 | 24 GB | GDDR5 | ~12 (FP32) | ~$0.07 |
| Tesla P4 | 8 GB | GDDR5 | ~5.5 (FP32) | ~$0.02 |

### Workstation / Quadro

| GPU | VRAM | Memory Type |  Price/hr |
|---|---|---|---|
| Quadro P4000 | 8 GB | GDDR5 | ~$0.03 |

### Consumer GeForce GTX 10xx

| GPU | VRAM | Memory Type |  Price/hr |
|---|---|---|---|
| GTX 1080 Ti | 11 GB | GDDR5X | ~$0.04 |
| GTX 1080 | 8 GB | GDDR5X | ~$0.03 |
| GTX 1070 Ti | 8 GB | GDDR5 | ~$0.04 |
| GTX 1070 | 8 GB | GDDR5 | ~$0.02 |
| GTX 1060 | 6 GB | GDDR5 | ~$0.09 |
| Titan Xp | 12 GB | GDDR5X | ~$0.04 |

---

> **Note:** Prices on  are set by individual hosts and fluctuate with supply and demand.  
> Always check pricing from respective provider for live rates before provisioning.