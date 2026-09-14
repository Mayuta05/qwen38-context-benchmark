# Qwen3.8-27B Long-Context Benchmark Corpus

This repository contains the fixed non-repetitive text corpus used for my
Qwen3.8-27B long-context benchmarks with llama.cpp.

The purpose of this repository is to make the benchmark corpus publicly
available so that results can be reproduced and compared across different
GPU configurations.

## Files

- `qwen-bench-master.txt`
  - The exact master corpus used for the benchmark.
  - Contains approximately 260K tokens when tokenized with Qwen3.8-27B.
  - Contains a mixture of prose, dialogue, code, tables, and logs.

- `make_qwen_bench_corpus.py`
  - The deterministic script used to generate the master corpus.
  - Random seed: `20260914`

## Benchmark Method

All context-length tests use the same master corpus.

Each test uses the first N tokens of `qwen-bench-master.txt`:

| Test | Context tokens |
|---|---:|
| 4K | 4,096 |
| 32K | 32,768 |
| 64K | 65,536 |
| 128K | 131,072 |
| 196K | 196,608 |
| 260K | 260,000 |

The corpus content is therefore shared between all tests:

`4K ⊂ 32K ⊂ 64K ⊂ 128K ⊂ 196K ⊂ 260K`

For each context length:

1. Start/restart `llama-server`.
2. Use the first N tokens of the master corpus as the prompt.
3. Generate 1,000 tokens.
4. Record prefill speed, decode speed, MTP acceptance, and mean draft length.
5. Wait until both GPUs cool down to ≤50°C before the next run.

Each data point is currently a single run (no averaging).

## Model

- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf`
- llama.cpp: b10935
- Commit: `8e330954a`
- KV cache: Q8_0 (K/V)
- Flash Attention: enabled
- MTP: `n_max = 2`
- Maximum context: 262,144 tokens

## llama-server Command

```bash
/path/to/llama.cpp/build/bin/llama-server \
  -m /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --device CUDA0,CUDA1 \
  --split-mode tensor \
  -ngl all \
  -fa on \
  --jinja \
  -c 262144 \
  --parallel 1 \
  -t 8 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --reasoning-effort low \
  --reasoning-preserve \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --load-mode mlock \
  --spec-draft-device CUDA0,CUDA1


RTX 3090 ×2 Results

Test system:

2× NVIDIA GeForce RTX 3090 24GB
NVLink connected
CUDA P2P enabled
AMD Ryzen Threadripper PRO 3945WX
128GB RDIMM
Ubuntu 22.04
No GPU power limit (~350W per GPU / ~700W combined under load)
Context	Prefill (tok/s)	Decode (tok/s)	MTP acceptance	Mean draft
4K	1673.51	116.36	91.23% (645/707)	2.82
32K	1696.19	92.85	77.75% (608/782)	2.55
64K	1516.61	86.28	86.34% (632/732)	2.73
128K	1228.07	73.89	95.77% (656/685)	2.91
196K	1026.23	62.23	97.63% (660/676)	2.95
260K	899.47	53.33	95.91% (656/684)	2.92

Decode speed decreased from 116.36 tok/s at 4K to
53.33 tok/s at 260K, a reduction of approximately 54.2%.

The 260K-token prompt plus 1,000 generated tokens completed successfully
on the two 24GB RTX 3090 GPUs.

More Results Coming

I am also testing the exact same benchmark on:

2× NVIDIA Tesla V100 PCIe 32GB
PCIe P2P enabled
No NVLink

The V100 results will be added here for direct comparison.

Reproducing the Benchmark

If you run this benchmark on another GPU configuration, feel free to share
your results.

Using the same corpus, context lengths, model, llama.cpp version, and
generation settings will make hardware comparisons much more meaningful.
