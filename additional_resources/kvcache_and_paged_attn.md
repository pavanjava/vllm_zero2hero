# KV Cache & PagedAttention — A Complete Study Guide

> Based on the vLLM architecture. Written for clarity with full worked examples.

---

## Table of Contents

1. [Why KV Cache Exists](#1-why-kv-cache-exists)
2. [What is KV Cache?](#2-what-is-kv-cache)
3. [The Problem with Traditional KV Cache](#3-the-problem-with-traditional-kv-cache)
4. [PagedAttention — The Solution](#4-pagedattention--the-solution)
5. [Key Terminology — Blocks, Pages, Slots](#5-key-terminology--blocks-pages-slots)
6. [The Block Allocator](#6-the-block-allocator)
7. [Slot Mapping — Full Worked Example](#7-slot-mapping--full-worked-example)
8. [Continuous Batching with Multiple Sequences](#8-continuous-batching-with-multiple-sequences)
9. [Decode Phase — How New Tokens Are Added](#9-decode-phase--how-new-tokens-are-added)
10. [CUDA Blocks vs CPU Blocks](#10-cuda-blocks-vs-cpu-blocks)
11. [Maximum Concurrency](#11-maximum-concurrency)
12. [vLLM Scheduler & KV Cache Manager Interaction](#12-vllm-scheduler--kv-cache-manager-interaction)
13. [Summary Cheatsheet](#13-summary-cheatsheet)

---

## 1. Why KV Cache Exists

In a Transformer, during the attention mechanism, for every new token generated, the model needs to attend over **all previous tokens**. Without caching, the model would recompute the Key (K) and Value (V) vectors for every past token at every step.

```
Without KV Cache (step 5 of generation):
  Recompute K,V for token 1
  Recompute K,V for token 2
  Recompute K,V for token 3
  Recompute K,V for token 4
  Compute  K,V for token 5  ← new token
```

This is extremely wasteful. KV Cache solves this by **storing** K and V once they are computed, so they can be reused.

```
With KV Cache (step 5 of generation):
  Load K,V for tokens 1-4 from cache  ← reuse
  Compute K,V for token 5             ← only this is new
```

---

## 2. What is KV Cache?

For each token in a sequence, at each Transformer layer, the model computes:
- **K** (Key vector) — what this token "offers"
- **V** (Value vector) — what information this token holds

These are stored in GPU memory so they don't need to be recomputed.

```
Sentence: "today we are going to learn"

After prefill, KV Cache contains:
  Token 1 "today"   → K1, V1
  Token 2 "we"      → K2, V2
  Token 3 "are"     → K3, V3
  Token 4 "going"   → K4, V4
  Token 5 "to"      → K5, V5
  Token 6 "learn"   → K6, V6
```

During decode, when the model generates token 7 (e.g., "machine"), it only computes K7, V7 and attends over the cached K1–K6.

---

## 3. The Problem with Traditional KV Cache

Traditional systems pre-allocate a **single large contiguous block** of GPU memory per request, sized for the maximum possible sequence length.

```
max_seq_len = 2048 tokens

Sequence 1 (actual: 200 tokens):
[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 used=200      wasted=1848 tokens worth of memory

Sequence 2 (actual: 500 tokens):
[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 used=500      wasted=1548 tokens

█ = used memory    ░ = wasted (pre-allocated but unused)
```

**Problems:**
- **Internal fragmentation** — memory reserved but unused within each allocation
- **External fragmentation** — after many requests complete, free memory exists in scattered chunks too small for new requests
- **Low GPU utilization** — studies show only 20–38% of allocated KV cache memory is actually used
- **Low concurrency** — fewer requests can run simultaneously

---

## 4. PagedAttention — The Solution

PagedAttention is inspired by **OS virtual memory and paging**. Instead of one giant contiguous allocation, memory is divided into small fixed-size **blocks (pages)**, and each request is assigned only the blocks it actually needs — allocated on demand as the sequence grows.

```
OS Concept          →    PagedAttention Equivalent
────────────────────────────────────────────────────
Page                →    Block
Page Size           →    Block Size (N tokens per block)
Page Frame          →    Physical Block (in GPU VRAM)
Virtual Address     →    Logical Block Index
Page Table          →    Block Table
Free Page Pool      →    Free Block Pool
Process             →    Sequence / Request
```

**Key insight:** Blocks do NOT need to be contiguous in GPU memory. A sequence's logical blocks are mapped to physical blocks via a **block table**, just like OS virtual memory maps virtual pages to physical frames.

---

## 5. Key Terminology — Blocks, Pages, Slots

These three terms confuse most people. Here is the definitive clarification:

### Blocks = Pages (Same Thing, Different Names)

| Term | Used By | Meaning |
|------|---------|---------|
| **Page** | OS literature, original paper | Fixed-size memory unit |
| **Block** | vLLM codebase and docs | Same fixed-size memory unit |

They are **identical concepts**. The PagedAttention paper uses "page" (OS analogy); vLLM code uses "block" (e.g., `block_size`, `num_cuda_blocks`).

### Slots — One Level Deeper

```
Block (Page)
  └── contains N Slots     (where N = block_size)
            └── each Slot holds KV vectors for exactly 1 Token
```

**Example with `block_size = 4`:**

```
┌──────────────────────────────────┐
│           Physical Block 3        │
│  ┌────────┬────────┬────────┬────────┐ │
│  │ Slot 12 │ Slot 13 │ Slot 14 │ Slot 15 │ │
│  │ "today" │  "we"   │  "are"  │ "going" │ │
│  └────────┴────────┴────────┴────────┘ │
└──────────────────────────────────┘

Block 3 occupies slots: 12, 13, 14, 15
```

### The Slot Formula

```
slot_number = (block_id × block_size) + offset_within_block

Example (block_size = 4):
  Block 3, position 0  →  3 × 4 + 0  =  slot 12
  Block 3, position 1  →  3 × 4 + 1  =  slot 13
  Block 3, position 2  →  3 × 4 + 2  =  slot 14
  Block 3, position 3  →  3 × 4 + 3  =  slot 15

  Block 4, position 0  →  4 × 4 + 0  =  slot 16
  Block 4, position 2  →  4 × 4 + 2  =  slot 18
```

---

## 6. The Block Allocator

The **Block Allocator** (inside the KV Cache Manager) manages all physical blocks in GPU memory. It maintains:

```
Free Block Pool:  [blk_2, blk_5, blk_6, blk_9, blk_11, blk_12, ...]
                            (a linked list of available physical blocks)
```

**Allocation lifecycle:**

```
New request arrives
      ↓
Scheduler calls allocate_slots()
      ↓
Block Allocator pops free blocks from pool
      ↓
Assigns them to the request's block table
      ↓
Request runs (prefill + decode)
      ↓
Request completes → blocks returned to free pool
```

**Block Table per sequence (example):**

```
Sequence "today we are going to learn" (6 tokens, block_size=2):

Logical Block │ Physical Block │ Slots Used
─────────────────────────────────────────
      0       │   Block 7      │  14, 15
      1       │   Block 3      │   6,  7
      2       │   Block 11     │  22, 23

Note: Physical blocks can be ANY free block — not necessarily contiguous!
```

The attention kernel uses this block table to locate and fetch KV vectors for each token during attention computation.

---

## 7. Slot Mapping — Full Worked Example

### Setup

```
Sentence:   "today  we  are  going  to  learn"
Tokens:       t1    t2   t3    t4   t5    t6
block_size = 4
```

**allocate_slots() assigns blocks sequentially:**

```
ceil(6 tokens / 4 per block) = 2 blocks needed
  → Block 1 allocated  (slots 4, 5, 6, 7)
  → Block 2 allocated  (slots 8, 9, 10, 11)
```

*(Block 0 slots 0–3 may be reserved for special tokens like BOS)*

### Block Layout

```
Block 1  (slots 4–7):
┌──────────┬──────────┬──────────┬──────────┐
│  Slot 4  │  Slot 5  │  Slot 6  │  Slot 7  │
│ "today"  │   "we"   │  "are"   │ "going"  │
└──────────┴──────────┴──────────┴──────────┘

Block 2  (slots 8–11):
┌──────────┬──────────┬──────────┬──────────┐
│  Slot 8  │  Slot 9  │ Slot 10  │ Slot 11  │
│   "to"   │ "learn"  │  (empty) │  (empty) │
└──────────┴──────────┴──────────┴──────────┘
  ← used ──────────→  ← internal fragment →
```

### Slot Mapping Array

```
Token:        "today"  "we"  "are"  "going"  "to"  "learn"
Position:        0      1      2       3       4      5
Slot:            4      5      6       7       8      9

slot_mapping = [4, 5, 6, 7, 8, 9]
```

This array is passed to the **PagedAttention kernel** during inference. For every token in the flattened super-sequence, the kernel reads its slot number and knows exactly where to write/read that token's KV in physical GPU memory.

---

## 8. Continuous Batching with Multiple Sequences

### Example Setup

```
3 prompts sent together:
  Seq 1: [1, 2, 3, 4, 5]          →  5 tokens
  Seq 2: [1, 6, 5, 7, 8, 9, 10]  →  7 tokens
  Seq 3: [1, 12, 13]              →  3 tokens

block_size = 4
```

### Step 1 — Continuous Batching Flattens Everything

```
input_ids = [1,2,3,4,5, 1,6,5,7,8,9,10, 1,12,13]
positions  = [0,1,2,3,4, 0,1,2,3,4,5,6,  0, 1, 2]
```

All sequences become a single "super sequence" fed to the GPU in one forward pass.

### Step 2 — Block Allocation

```
Seq 1 needs ceil(5/4) = 2 blocks  →  Block 1, Block 2
Seq 2 needs ceil(7/4) = 2 blocks  →  Block 3, Block 4
Seq 3 needs ceil(3/4) = 1 block   →  Block 5
```

Blocks are assigned first-come-first-served.

### Step 3 — Physical Memory Layout

```
GPU VRAM:
┌──────┬──────┬──────┬──────┬──────┐
│ blk1 │ blk2 │ blk3 │ blk4 │ blk5 │
└──────┴──────┴──────┴──────┴──────┘
  Seq1   Seq1   Seq2   Seq2   Seq3
```

### Step 4 — Slot Mapping Computed

```
Block 1 → slots  4,  5,  6,  7
Block 2 → slots  8,  9, 10, 11
Block 3 → slots 12, 13, 14, 15
Block 4 → slots 16, 17, 18, 19
Block 5 → slots 20, 21, 22, 23

Seq 1 (5 tokens):  slots → [4, 5, 6, 7, 8]
Seq 2 (7 tokens):  slots → [12, 13, 14, 15, 16, 17, 18]
Seq 3 (3 tokens):  slots → [20, 21, 22]

Full slot_mapping = [4,5,6,7,8, 12,13,14,15,16,17,18, 20,21,22]
```

**Why does Seq 2 start at slot 12?**
- Seq 1 consumed Block 1 (slots 4–7) and Block 2 (slot 8 partial)
- Next available block is Block 3 → starts at `3 × 4 = 12`

**Why is Block 4 only 75% full for Seq 2?**
- Seq 2 has 7 tokens: 4 go into Block 3 (full), 3 go into Block 4
- `3/4 = 75%` of Block 4 used → 1 slot of internal fragmentation (unavoidable, bounded)

### Step 5 — Attention Metadata

```
query_start_loc = [0, 5, 12]    ← where each seq starts in super-sequence
seq_lens        = [5, 7, 3]     ← length of each sequence
num_actual_tokens = 15          ← total tokens in this batch
```

The attention kernel uses `slot_mapping` + `query_start_loc` to correctly compute attention for each sequence independently, even though they're batched together.

---

## 9. Decode Phase — How New Tokens Are Added

After prefill, the model generates new tokens one at a time.

### Continuing the Example

**Sampled tokens:** [14, 15, 16] — one new token per sequence

**Continuous batching appends them:**
```
input_ids = [1,2,3,4,5,14, 1,6,5,7,8,9,10,15, 1,12,13,16]
positions  = [0,1,2,3,4,5,  0,1,2,3,4,5,6, 7,  0, 1, 2, 3]
```

**New slot_mapping (only new tokens need new slots):**
```
Token 14 for Seq 1 → goes into slot 9  (Block 2, position 1)
Token 15 for Seq 2 → goes into slot 19 (Block 4, position 3)
Token 16 for Seq 3 → goes into slot 23 (Block 5, position 3)

new slot_mapping = [4,5,6,7,8,9, 12,13,14,15,16,17,18,19, 20,21,22,23]
```

**Key optimization:** During decode, vLLM **reuses** the KV vectors from prefill (already stored in their slots). It only computes KV for the 1 new token per sequence — not the entire history. This is the core efficiency of KV Cache + PagedAttention working together.

```
Decode step cost:
  Old systems: O(total_tokens²) — recompute everything
  vLLM:        O(new_tokens)    — only compute 1 KV per sequence
```

---

## 10. CUDA Blocks vs CPU Blocks

When vLLM starts, it logs:

```
10:57:20 [executor_base.py:114] # cuda blocks: 32357  # cpu blocks: 11915
```

| | CUDA Blocks | CPU Blocks |
|---|---|---|
| Location | GPU VRAM | System RAM |
| Speed | Very fast (nanoseconds) | Slow (PCIe transfer, microseconds) |
| Purpose | Active KV cache for running sequences | Swap space for preempted sequences |
| Who manages | KV Cache Manager | KV Cache Manager |

**How CUDA blocks are calculated:**

```
cuda_blocks = (Total VRAM − model_weights − activation_memory − overhead)
              / size_of_one_block

size_of_one_block = block_size × num_heads × head_dim × 2 (K and V) × dtype_bytes
```

**Swapping flow:**

```
GPU VRAM full (no free CUDA blocks)?
         ↓
Preempt lowest priority sequence
         ↓
Swap its KV blocks → CPU RAM (CPU blocks)
         ↓
Free CUDA blocks now available for new/higher priority sequence
         ↓
When preempted sequence is rescheduled → swap back to GPU
```

---

## 11. Maximum Concurrency

```
10:57:20 [executor_base.py:119] Maximum concurrency for 2048 tokens per request: 252.79x
```

**Formula:**

```
blocks_per_request = ceil(max_tokens / block_size)
                   = ceil(2048 / 16)
                   = 128 blocks

max_concurrency = total_cuda_blocks / blocks_per_request
               = 32357 / 128
               = 252.79x
```

**Concurrency scales inversely with sequence length:**

| Tokens per Request | Blocks Needed | Max Concurrent Requests |
|---|---|---|
| 512  |  32 | ~1011 |
| 1024 |  64 |  ~505 |
| 2048 | 128 |  ~252 |
| 4096 | 256 |  ~126 |

> Longer sequences = fewer concurrent users. This is why `max_model_len` is an important serving configuration parameter.

---

## 12. vLLM Scheduler & KV Cache Manager Interaction

The vLLM Scheduler runs every step and does the following:

```
┌─────────────────────────────────────────────────────┐
│                   vLLM Scheduler                     │
│                                                      │
│  1. Prioritize decode requests                       │
│     (sequences already in "running queue" first)     │
│                                                      │
│  2. Compute number of new tokens to generate         │
│                                                      │
│  3. Call KV Cache Manager's allocate_slots()         │
│     → Block Allocator assigns physical blocks        │
│     → slot_mapping is computed                       │
│                                                      │
│  4. Update token budget                              │
│     token_budget -= num_new_tokens                   │
│                                                      │
│  5. Process prefill requests from waiting queue      │
│     (if budget remaining)                            │
└─────────────────────────────────────────────────────┘
```

**Queue states for a sequence:**

```
Waiting Queue    →    Running Queue    →    Done
   (prefill           (decode phase,         ↓
   pending)           has KV cache)      blocks freed
                                         back to pool
        ↑                  |
        └──── preempted ───┘
              (swapped to CPU)
```

---

## 13. Summary Cheatsheet

### Terminology

| Term | Definition |
|------|-----------|
| **KV Cache** | Stored Key-Value vectors for all past tokens, per layer |
| **Block / Page** | Fixed-size container in GPU memory (same thing, different names) |
| **block_size** | Number of token slots per block (commonly 16) |
| **Slot** | One position inside a block; holds KV for exactly 1 token |
| **Block Table** | Per-sequence mapping: logical block → physical block |
| **slot_mapping** | Array telling the attention kernel which physical slot each token uses |
| **Free Block Pool** | Linked list of unallocated physical blocks |
| **Block Allocator** | Component that manages the free block pool |
| **CUDA Blocks** | Physical blocks in GPU VRAM (fast, primary) |
| **CPU Blocks** | Physical blocks in system RAM (slow, swap space) |

### Key Formulas

```python
# Slot number from block info
slot = block_id * block_size + offset_within_block

# Blocks needed for a sequence
blocks_needed = ceil(num_tokens / block_size)

# Maximum concurrent requests
max_concurrency = total_cuda_blocks / ceil(max_tokens / block_size)

# Total CUDA blocks available
cuda_blocks = (VRAM - weights - activations - overhead) / block_size_bytes
```

### The Full Pipeline (One Line Each)

```
1. Request arrives          → added to waiting queue
2. Scheduler picks it up    → calls allocate_slots()
3. Block Allocator          → assigns physical blocks, builds block table
4. slot_mapping computed    → [slot_for_token_0, slot_for_token_1, ...]
5. Prefill forward pass     → KV computed and written to assigned slots
6. Decode loop              → 1 new token/step, 1 new slot allocated per step
7. Request completes        → all blocks returned to free pool
```

---

*This document covers the complete KV Cache and PagedAttention theory as implemented in vLLM, grounded in the continuous batching example with sequences "today we are going to learn..." and the multi-sequence slot mapping walkthrough.*