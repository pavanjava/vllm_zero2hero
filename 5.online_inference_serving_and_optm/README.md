# Online serving (OpenAI Compatible)

### Launch vLLM with prefix cache, chunked-prefill, quantization
```commandline
vllm serve google/gemma-4-31B-it \
  --port 8000 \
  --max-model-len ${VLLM_MAX_MODEL_LEN:-32768} \
  --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE:-2} \
  --max-num-seqs ${VLLM_MAX_NUM_SEQS:-32} \
  --reasoning-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTIL:-0.90} \
  --kv-cache-dtype auto \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3", "num_speculative_tokens":4, "method": "eagle3"}' \
  --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS:-8192} \
  --compilation-config '{"mode":3}' \
  --quantization fp8 \
  --enable-prompt-tokens-details
```

### Integrate vLLM with LMCache
```commandline
# Step 1 — Install LMCache
pip install lmcache

# Step 2 — Start the LMCache server (standalone MP mode, recommended for production)
# L1 CPU cache: 20 GB | eviction: LRU | chunk size: 256 tokens (production default)
lmcache server \
  --l1-size-gb ${LMCACHE_L1_SIZE_GB:-20} \
  --eviction-policy LRU \
  --chunk-size ${LMCACHE_CHUNK_SIZE:-256}

# Step 3 — Launch vLLM with LMCache MP connector (in a separate terminal)
vllm serve google/gemma-4-31B-it \
  --port 8000 \
  --max-model-len ${VLLM_MAX_MODEL_LEN:-32768} \
  --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE:-2} \
  --max-num-seqs ${VLLM_MAX_NUM_SEQS:-32} \
  --reasoning-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTIL:-0.90} \
  --kv-cache-dtype auto \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3", "num_speculative_tokens":4, "method": "eagle3"}' \
  --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS:-8192} \
  --compilation-config '{"mode":3}' \
  --quantization fp8 \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector", "kv_role":"kv_both"}'
```

**Why MP connector?**
- Runs LMCache as a standalone service; vLLM attaches via ZMQ (default port `5555`)
- Supports sharing one KV cache across multiple vLLM engine instances
- Ships fixes ahead of the version vendored into vLLM — always tracks latest protocol
- HTTP management/metrics endpoint exposed at `localhost:8080`