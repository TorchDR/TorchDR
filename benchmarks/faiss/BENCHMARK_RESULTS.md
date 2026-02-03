# FAISS Index Benchmark Results

Benchmarks run on NVIDIA B200 GPU with 128-dimensional vectors, k=15 neighbors.

## 1M Samples - Clustered Data (1000 Gaussian clusters)

| Index Type | nlist | nprobe | Time (s) | Recall@k |
|------------|-------|--------|----------|----------|
| Flat (exact) | - | - | 10.19 | 100.0% |
| IVF | 4096 | 40 | 2.90 | 99.9% |
| IVF | 4096 | 204 | 6.64 | 99.9% |
| IVF | 1024 | 10 | 3.81 | 99.9% |
| IVFPQ (M=16) | 4096 | 40 | 3.84 | 33.2% |
| IVFPQ (M=16) | 1024 | 10 | 2.96 | 18.2% |

**Key insight**: On clustered data, IVF achieves 99.9% recall with 3.5x speedup using minimal nprobe.

## 1M Samples - Random Data (uniform)

| Index Type | nlist | nprobe | Time (s) | Recall@k |
|------------|-------|--------|----------|----------|
| Flat (exact) | - | - | 10.16 | 100.0% |
| IVF | 1024 | 10 | 2.56 | 10.1% |
| IVF | 1024 | 102 | 11.14 | 44.9% |
| IVF | 1024 | 512 | 50.70 | 91.1% |
| IVFPQ (M=16) | 1024 | 10 | 3.54 | 4.6% |
| IVFPQ (M=16) | 1024 | 512 | 36.24 | 11.6% |

**Key insight**: On random data, IVF needs high nprobe (50% of nlist) to achieve >90% recall.

## 10M Samples - Clustered Data (1000 Gaussian clusters)

| Index Type | nlist | nprobe | Time (s) | Recall@k |
|------------|-------|--------|----------|----------|
| IVF | 16384 | 81 | 54.69 | 99.9% |
| IVF | 16384 | 409 | 271.32 | 99.9% |
| IVF | 65536 | 327 | 143.25 | 99.9% |
| IVFPQ (M=16) | 16384 | 81 | 47.02 | 22.3% |
| IVFPQ (M=16) | 65536 | 327 | 262.37 | 23.2% |

**Key insight**: IVF maintains 99.9% recall at 10M scale with low nprobe on clustered data.

## 10M Samples - Random Data (uniform)

| Index Type | nlist | nprobe | Time (s) | Recall@k |
|------------|-------|--------|----------|----------|
| IVF | 16384 | 81 | 84.95 | 12.6% |
| IVF | 16384 | 409 | 320.59 | 32.8% |
| IVF | 16384 | 819 | 610.63 | 46.3% |
| IVFPQ (M=16) | 16384 | 81 | 57.52 | 4.4% |
| IVFPQ (M=16) | 16384 | 819 | 957.79 | 7.7% |

**Key insight**: Random data requires much higher nprobe for decent recall. IVFPQ performs poorly.

## Summary

| Data Type | Best Config for >95% Recall | Speedup vs Flat |
|-----------|----------------------------|-----------------|
| 1M Clustered | IVF nlist=4096 nprobe=40 | 3.5x |
| 1M Random | IVF nlist=1024 nprobe=512 | 0.2x (slower) |
| 10M Clustered | IVF nlist=16384 nprobe=81 | N/A* |
| 10M Random | Not achieved (max 46.3%) | N/A* |

*Flat baseline not computed for 10M (too slow).

## Recommendations

1. **For clustered/real-world data**: Use IVF with low nprobe (0.5-1% of nlist) for excellent recall with significant speedup.

2. **For random/uniform data**: IVF provides limited benefit. Consider using Flat index or much higher nprobe.

3. **IVFPQ**: Best for memory-constrained scenarios. Recall is limited by PQ compression (~20-30% on clustered data).

4. **nlist selection**: Follow FAISS guidelines: `4*sqrt(n)` to `16*sqrt(n)` where n is dataset size.
