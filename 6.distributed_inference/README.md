# run_cluster.sh

![bash](https://img.shields.io/badge/shell-bash-lightgrey) ![Docker](https://img.shields.io/badge/runtime-Docker-blue) ![Ray](https://img.shields.io/badge/cluster-Ray-green) ![vLLM](https://img.shields.io/badge/inference-vLLM-purple)

Launch a multi-node Ray cluster inside Docker for distributed vLLM inference. Designate one machine as the head node and connect any number of worker nodes to serve large language models across a GPU fleet.

---

## Table of contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Arguments](#arguments)
- [Node roles](#node-roles)
- [Container management](#container-management)
- [Docker flags](#docker-flags)
- [Important notes](#important-notes)

---

## Overview

The script abstracts the complexity of wiring a Ray cluster across multiple machines. It handles container naming, Ray start command assembly, network configuration, and automatic cleanup — so you can focus on serving models, not cluster plumbing.

---

## Prerequisites

The following must be installed and available on every machine in the cluster before running this script:

| Requirement | Details |
|---|---|
| **Docker** | With GPU support enabled (`nvidia-container-toolkit` for NVIDIA GPUs). Must be accessible without sudo, or run as root. |
| **NVIDIA drivers** | Required for `--gpus all` to expose GPUs to the container. Verify with `nvidia-smi`. |
| **Network reachability** | All worker machines must reach the head node IP on port `6379`. Firewall rules must permit this traffic. |
| **HuggingFace cache** | An absolute path to a directory where model weights are cached. Mounted into the container to avoid re-downloading models. |

---

## Quick start

### 1 — Start the head node

Run this on the machine that will coordinate the cluster:

```bash
# On the head machine
bash run_cluster.sh \
    vllm/vllm-openai \
    <head_node_ip> \
    --head \
    /abs/path/to/huggingface/cache \
    -e VLLM_HOST_IP=<head_node_ip>
```

### 2 — Start each worker node

Run this on every additional GPU machine. Replace `VLLM_HOST_IP` with that machine's own IP address:

```bash
# On each worker machine
bash run_cluster.sh \
    vllm/vllm-openai \
    <head_node_ip> \
    --worker \
    /abs/path/to/huggingface/cache \
    -e VLLM_HOST_IP=<worker_node_ip>
```

### 3 — Serve a model

Once the cluster is running, exec into the head node container and issue vLLM serve commands as if you were on a single machine:

```bash
# Find your container name (printed on startup)
docker exec -it node-<random_suffix> /bin/bash

# Inside the container
vllm serve <model_id> --tensor-parallel-size <N>
```

---

## Arguments

| Position | Name | Description |
|---|---|---|
| `$1` | `docker_image` | The Docker image to run. Typically `vllm/vllm-openai` or a custom image built on top of it. |
| `$2` | `head_node_ip` | IP address of the head node. Workers use this to connect to the Ray cluster on port `6379`. |
| `$3` | `--head` \| `--worker` | Node role flag. Exactly one must be provided. Controls whether Ray starts a new cluster or joins an existing one. |
| `$4` | `path_to_hf_home` | Absolute path to the HuggingFace model cache on the host. Mounted to `/root/.cache/huggingface` inside the container. |
| `$5+` | `[additional_args]` | Extra flags passed directly to `docker run`. Commonly used to inject environment variables, e.g. `-e VLLM_HOST_IP=...`. Each worker must supply its own unique IP here. |

---

## Node roles

### Head node — cluster coordinator

- Starts Ray with `--head`, binds to port `6379`, and manages cluster state.
- Run exactly **one** per cluster.
- `VLLM_HOST_IP` must match `head_node_ip`. If they differ, the script issues a warning and uses `VLLM_HOST_IP` as the authoritative address.

### Worker node — GPU contributor

- Connects to the head via `--address=<head_ip>:6379`.
- Contributes all GPUs on the host machine to the shared resource pool.
- Each worker must be given its own machine's IP via `-e VLLM_HOST_IP=<worker_ip>`.

---

## Container management

Each invocation generates a unique container name in the format `node-<random>`, allowing multiple Ray containers to coexist on the same machine (useful on multi-GPU hosts).

| Action | Command |
|---|---|
| Exec into container | `docker exec -it node-<random_suffix> /bin/bash` |
| Stop container manually | `docker stop node-<random_suffix>` |
| Auto-cleanup | A `trap cleanup EXIT` hook automatically stops and removes the container when the script terminates, preventing orphaned containers. |

---

## Docker flags

| Flag | Purpose |
|---|---|
| `--network host` | Uses the host network stack so Ray nodes communicate directly without NAT or port mapping overhead. |
| `--shm-size 10.24g` | Allocates 10.24 GB of shared memory. Required for inter-process communication in multi-GPU tensor-parallel inference. |
| `--gpus all` | Exposes all host GPUs to the container via the NVIDIA container runtime. |
| `-v HF_HOME` | Mounts the HuggingFace cache directory so model weights are reused across runs without re-downloading. |

---

## Important notes

> **Keep terminal sessions open.**
> Closing the terminal that launched the script triggers the EXIT trap, which stops and removes the container. This shuts down the associated Ray node and can bring down the entire cluster if it is the head node.

> **VLLM_HOST_IP uniqueness.**
> Every worker must be given its own machine's IP via `-e VLLM_HOST_IP=<worker_ip>`. Reusing the head node IP on workers will cause Ray to misroute traffic.

> **Head node IP consistency.**
> If `VLLM_HOST_IP` is provided on the head node and differs from `head_node_ip`, the script issues a warning and uses `VLLM_HOST_IP` as the authoritative address.