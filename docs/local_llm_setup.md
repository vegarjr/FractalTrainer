# Local LLM setup (offline, GTX 1050)

This document records the local-LLM teacher setup for this specific
machine so it's reproducible and so future-you knows the context.

## The stack

- **Model:** Qwen2.5-Coder-7B-Instruct (Q4_K_M quantization, ~4.68 GB)
- **Runner:** llama.cpp (b8857 release, Vulkan x64 build)
- **Hardware:** NVIDIA GeForce GTX 1050, 4 GB VRAM
- **Acceleration:** Vulkan (CUDA binaries aren't shipped for Linux
  anymore; Vulkan works on NVIDIA hardware via the standard driver)
- **Observed speed:** ~7–8 tokens/sec at 20/28 layers on GPU, 4K context

## One-time install

```bash
# 1. Get llama.cpp Vulkan build (~33 MB):
mkdir -p ~/.local/llama.cpp ~/.local/models
cd /tmp
curl -sL -o llama-vulkan.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b8857/llama-b8857-bin-ubuntu-vulkan-x64.tar.gz
tar -xzf llama-vulkan.tar.gz -C ~/.local/llama.cpp

# 2. Get Qwen2.5-Coder-7B-Instruct Q4_K_M (~4.68 GB):
curl -L -o ~/.local/models/qwen25-coder-7b.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

## Start the server

```bash
export LD_LIBRARY_PATH=$HOME/.local/llama.cpp/llama-b8857:$LD_LIBRARY_PATH
~/.local/llama.cpp/llama-b8857/llama-server \
  -m ~/.local/models/qwen25-coder-7b.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 20 --ctx-size 4096 &
```

Takes ~10 seconds to load into VRAM. Health-check:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"ok"}
```

Stop with `kill $(pgrep llama-server)`.

## Use from FractalTrainer

The repair-loop has a fourth LLM backend alongside `mock`, `cli`, `api`:

```bash
python scripts/run_closed_loop.py --llm local
# Uses http://127.0.0.1:8080 by default; override with --local-llm-url
```

From Python:

```python
from fractaltrainer.repair import make_local_llm_client

fn = make_local_llm_client()  # defaults to 127.0.0.1:8080, Qwen-labelled
response = fn("You are a concise code assistant.",
              "Write a Python BFS function.")
```

Signature matches `make_claude_cli_client` and `make_claude_client` —
the repair loop accepts any of the three interchangeably.

## VRAM tuning on this hardware

| --n-gpu-layers | VRAM usage | Speed (steady-state) |
|---:|---:|---:|
| 0  | ~80 MB | ~3 tok/s (CPU only) |
| 10 | ~1.8 GB | ~5 tok/s |
| **20** | **~3.5 GB** | **~8 tok/s (current default)** |
| 28 | OOM on 4 GB VRAM | — |

`--n-gpu-layers 20` is the sweet spot on GTX 1050 + 4 GB. If you upgrade
the GPU, raise to 28 (full model on GPU) for 20+ tok/s.

## What we paid to set this up

- 17 GB disk (models + binaries — most was the original CUDA build
  attempt that didn't have a Linux binary, we fell back to Vulkan)
- ~10 min downloads + 2 min first model-load
- Zero API cost; zero latency to external services; data stays local.

## When to prefer local vs Claude CLI

- **Prefer `--llm local`** when: offline, privacy-sensitive, rapid
  iteration (hundreds of calls), experimenting with prompts where
  quality doesn't need to be state-of-the-art.
- **Prefer `--llm cli` (Claude)** when: need strong reasoning, code
  quality matters for a final artifact, one-off calls where the 7B
  model's ~40% HumanEval vs Claude's ~92% is the binding difference.
- **Both work interchangeably** via the same RepairLoop backend — you
  can prototype on local and promote to Claude for the final run.

## References

- Qwen2.5-Coder model card: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
- llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases
