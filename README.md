# vllm_zero2hero


### Which all inferencing providers are covered ?
- vLLM
- SGLang (SGL)
- Transformers serve
- Huggingface TGI
- Ollama

### Installation of vLLM
```shell
# step1: create virtual env
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate

#step2: upgrade pip
pip install --upgrade pip

# step3: install vLLM
pip install vllm

# step4: verify Installation
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# step5: serving model
vllm serve google/gemma-4-E4B --dtype bfloat16
or
vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 8000 --dtype bfloat16 --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.80 --max-model-len 16384 
  
```

### Installation of SGL
```shell
# step1: install system dependency first (critical - missing this breaks sgl_kernel)
apt-get update && apt-get install -y libnuma1 libnuma-dev

# step2: create virtual env
python3 -m venv /opt/sglang-env
source /opt/sglang-env/bin/activate

# step3: upgrade pip
pip install --upgrade pip

# step4: install SGLang
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/

# step5: verify sgl_kernel loads cleanly
python -c "import sgl_kernel; print('sgl_kernel OK')"

# step6: verify installation
python -c "import sglang; print(sglang.__version__)"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# step7: serving model
python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --dtype auto --port 30000
```

### Installation of Ollama
```shell
# step1: install Ollama (no venv needed, standalone binary)
curl -fsSL https://ollama.com/install.sh | sh

# step2: verify installation
ollama --version

# step3: verify GPU is detected
ollama info

# step4: pull and serve model
ollama pull gemma3:latest

# step5: serving model (starts server + loads model)
ollama serve &
ollama run gemma3:latest
```

### Load Test SGL vs vLLM
```shell
# install locust
pip install locust

# run load test: 10 users, 10 minutes, headless
locust -f locustfile.py \
  --headless \
  --users 10 \
  --spawn-rate 2 \
  --run-time 10m \
  --html report.html
```
```web
locust -f locustfile.py
# then open http://localhost:8089
# set: Users=10, Spawn rate=2, Host=https://uoum3yyi76wdqh-8000.proxy.runpod.net
```

### Spin Models on vLLM
```commandline
vllm serve google/gemma-4-31B-it \
  --max-model-len 16384 \
  --tensor-parallel-size 3 \
  --gpu-memory-utilization 0.9

```

### Licensing
MIT 