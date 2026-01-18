# Optimized Related Work Generation with Multi-Agent Systems

Minimal instructions to run the demo and benchmark.

## Requirements
- Python 3.10+
- NVIDIA GPU (tested on ~12GB VRAM)
- vLLM installed and a Qwen3-8B (quantized) model available locally

## Install

```bash
pip install -r requirements.txt
```

## Start vLLM (example)

```bash
vllm serve ./Qwen3-8B-quantized.w4a16 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --enforce-eager \
  --enable-reasoning \
  --reasoning-parser deepseek_r1
```

## Run demos

```bash
# Single-agent demo (arXiv retrieval)
python single_agent.py

# Multi-agent demo (arXiv retrieval)
python run_mas.py
```

## Run benchmark (MAS vs Single Agent on OARelatedWork)

```bash
# Compare MAS vs single agent on high-reference samples
python benchmark_optimized.py --samples 10 --min-refs 7 --compare --max-revisions 2
```

