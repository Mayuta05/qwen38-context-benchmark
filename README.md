# Qwen3.8-27B Long-Context Benchmark Corpus

This repository contains the fixed non-repetitive text corpus and generator script used for my Qwen3.8-27B long-context benchmarks with llama.cpp.

The purpose of this repository is to make the benchmark corpus publicly available so that results can be reproduced and compared across different GPU configurations.

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
2. Use the first N tokens of the same master corpus as the prompt.
3. Generate 1,000 tokens.
4. Record prefill speed, decode speed, MTP acceptance, and mean draft length.
5. Wait until both GPUs cool down to ≤50°C before starting the next run.

Each data point is a single run (no averaging).

This procedure was kept the same for both the RTX 3090 ×2 and Tesla V100 ×2 tests.

## Model and Software

- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf`
- llama.cpp: b10935
- Commit: `8e330954a`
- GPU split: `--split-mode tensor`
- KV cache: Q8_0 (K/V)
- Flash Attention: enabled
- MTP: `n_max = 2`
- Maximum context length: 262,144 tokens
- CPU threads: 8
- `mlock`: enabled

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
```

---

# RTX 3090 24GB ×2 Results

## Test System

- CPU: AMD Ryzen Threadripper PRO 3945WX
- Memory: 128GB RDIMM
- GPU: 2× NVIDIA GeForce RTX 3090 24GB
- GPU interconnect: NVLink
- CUDA P2P: enabled
- OS: Ubuntu 22.04
- No GPU power limit
- Approximately 350W per GPU / ~700W combined under load

## Results

| Context | Prefill (tok/s) | Decode (tok/s) | MTP acceptance | Mean draft |
|---:|---:|---:|---:|---:|
| 4,096 | 1673.51 | 116.36 | 91.23% (645/707) | 2.82 |
| 32,768 | 1696.19 | 92.85 | 77.75% (608/782) | 2.55 |
| 65,536 | 1516.61 | 86.28 | 86.34% (632/732) | 2.73 |
| 131,072 | 1228.07 | 73.89 | 95.77% (656/685) | 2.91 |
| 196,608 | 1026.23 | 62.23 | 97.63% (660/676) | 2.95 |
| 260,000 | 899.47 | 53.33 | 95.91% (656/684) | 2.92 |

Decode speed decreased from **116.36 tok/s at 4K** to **53.33 tok/s at 260K**, a reduction of approximately **54.2%**.

The **260K-token prompt + 1,000 generated tokens** completed successfully on the two 24GB RTX 3090 GPUs.

---

# Tesla V100 PCIe 32GB ×2 Results

## Test System

- CPU: AMD Ryzen Threadripper PRO 3945WX
- Memory: 128GB RDIMM
- GPU: 2× NVIDIA Tesla V100 PCIe 32GB
- PCIe: Gen3 x16 ×2
- CUDA P2P: enabled over PCIe
- NVLink: not used
- OS: Ubuntu 22.04

CUDA P2P between the two V100 GPUs was verified with NVIDIA's `p2pBandwidthLatencyTest`.

Measured V100-to-V100 P2P bandwidth:

- Unidirectional P2P enabled: ~13.16 GB/s
- Bidirectional P2P enabled: ~25.5 GB/s
- GPU P2P write latency: ~2.2 µs

## Results

| Context | Prefill (tok/s) | Decode (tok/s) | MTP acceptance | Mean draft |
|---:|---:|---:|---:|---:|
| 4,096 | 1039.95 | 82.90 | 92.03% (647/703) | 2.84 |
| 32,768 | 954.21 | 65.99 | 80.76% (617/764) | 2.62 |
| 65,536 | 805.91 | 56.77 | 79.40% (613/772) | 2.59 |
| 131,072 | 607.33 | 50.18 | 95.63% (656/686) | 2.91 |
| 196,608 | 485.28 | 37.96 | 85.60% (630/736) | 2.71 |
| 260,000 | 406.05 | 29.98 | 93.39% (650/696) | 2.87 |

Decode speed decreased from **82.90 tok/s at 4K** to **29.98 tok/s at 260K**, a reduction of approximately **63.8%**.

The **260K-token prompt + 1,000 generated tokens** also completed successfully on the two 32GB Tesla V100 PCIe GPUs.

---

# RTX 3090 ×2 vs Tesla V100 ×2

## Decode Performance

| Context | RTX 3090 ×2 | Tesla V100 ×2 | 3090 / V100 |
|---:|---:|---:|---:|
| 4K | 116.36 tok/s | 82.90 tok/s | 1.40× |
| 32K | 92.85 tok/s | 65.99 tok/s | 1.41× |
| 64K | 86.28 tok/s | 56.77 tok/s | 1.52× |
| 128K | 73.89 tok/s | 50.18 tok/s | 1.47× |
| 196K | 62.23 tok/s | 37.96 tok/s | 1.64× |
| 260K | 53.33 tok/s | 29.98 tok/s | 1.78× |

Both configurations show a substantial decrease in decode performance as the context becomes longer.

The RTX 3090 ×2 setup is faster across all tested context lengths, and the performance gap generally becomes larger at very long contexts.

At 260K:

- RTX 3090 ×2: **53.33 tok/s**
- Tesla V100 ×2: **29.98 tok/s**
- RTX 3090 ×2 is approximately **1.78× faster**

The interconnects are different between the two systems:

- RTX 3090 ×2: **NVLink + CUDA P2P**
- Tesla V100 ×2: **PCIe Gen3 x16 ×2 + CUDA P2P**

Because of this, these results should be considered a comparison of the complete real-world configurations rather than a pure GPU architecture comparison.

## Prefill Performance

| Context | RTX 3090 ×2 | Tesla V100 ×2 |
|---:|---:|---:|
| 4K | 1673.51 tok/s | 1039.95 tok/s |
| 32K | 1696.19 tok/s | 954.21 tok/s |
| 64K | 1516.61 tok/s | 805.91 tok/s |
| 128K | 1228.07 tok/s | 607.33 tok/s |
| 196K | 1026.23 tok/s | 485.28 tok/s |
| 260K | 899.47 tok/s | 406.05 tok/s |

---

# Notes About MTP

MTP acceptance is not monotonic with context length in these results.

This is expected because MTP acceptance is affected by the predictability and local content of the prompt, not just the context length.

The same fixed master corpus was therefore used for both GPU configurations so that the MTP workload would be as comparable as possible.

All context-length prompts are prefixes of the same master corpus.

---

# Reproducing the Benchmark

If you want to reproduce this benchmark:

1. Use `qwen-bench-master.txt`.
2. Tokenize it with the same Qwen3.8-27B model.
3. Use the first N tokens for the desired context length.
4. Use the llama.cpp configuration shown above.
5. Generate exactly 1,000 tokens.
6. Restart `llama-server` between context lengths.
7. Avoid prompt/KV reuse between tests.

For the measurements published here, both GPUs were allowed to cool down to ≤50°C before starting the next run.

If you run the same benchmark on another GPU configuration, feel free to share your results.

Results from other configurations such as RTX 4090, RTX A6000, A100, H100, other Tesla GPUs, mobile GPUs, or other multi-GPU systems would be very interesting to compare.

## Disclaimer

These results are intended as practical homelab measurements rather than standardized laboratory benchmarks.

Each data point is a single run and was not averaged across multiple runs.

MTP performance can vary depending on prompt content and draft-token acceptance, so using the same corpus is important when comparing results.
