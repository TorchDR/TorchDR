# TorchDR Affinity Benchmark Report

## Executive Summary

Benchmark of various affinity matrix computations on MNIST dataset (10,000 samples) using NVIDIA A100-80GB GPU. Tests compare runtime and memory usage across different precision modes (32-bit, 16-bit mixed, bfloat16 mixed).

## Test Environment

- **Hardware**: NVIDIA A100-SXM4-80GB
- **Software**: PyTorch 2.7.0+cu126
- **Dataset**: MNIST (10,000 samples × 784 features)
- **Data Size**: 29.91 MB input
- **Date**: 2025-08-08

## Key Findings

### 🏆 Performance Winners

#### Fastest Affinities (32-bit)
1. **Student-t**: 0.02s, 1176MB (dense output)
2. **Gaussian**: 0.02s, 1184MB (dense output)
3. **PACMAP**: 0.02s, 1176MB (lazy tensor)
4. **UMAP**: 0.04s, 1176MB (lazy tensor)
5. **DoublyStochasticQuad**: 0.04s, 1558MB (dense output)

#### Most Memory Efficient
1. **PACMAP**: 1176MB
2. **Student-t**: 1176MB
3. **Entropic**: 1176MB (but slower at 0.61s)
4. **UMAP**: 1176MB
5. **Gaussian**: 1184MB

### Mixed Precision Impact

Mixed precision (16-bit and bfloat16) showed **minimal benefits** for affinity computations:
- **Best speedup**: 1.79x (Gaussian with bf16)
- **Average speedup**: ~1.2-1.6x for fast operations
- **Memory reduction**: Negligible (0-0.7%)
- **Stability issues**: Several affinities fail with float16 (SymmetricEntropic, Sinkhorn, DoublyStochasticQuad)

## Detailed Results

### Dense Output Affinities

| Affinity | Runtime (32-bit) | Memory (32-bit) | 16-bit Speedup | bf16 Speedup | Status |
|----------|-----------------|-----------------|----------------|--------------|---------|
| **Gaussian** | 0.02s | 1184 MB | 1.77x ✓ | 1.79x ✓ | Stable |
| **Student-t** | 0.02s | 1176 MB | 1.61x ✓ | 1.60x ✓ | Stable |
| **SymmetricEntropic** | 3.03s | 1941 MB | Failed ✗ | 1.00x ✓ | Float16 unstable |
| **Sinkhorn** | 0.23s | 1666 MB | Failed ✗ | 1.03x ✓ | Float16 unstable |
| **DoublyStochasticQuad** | 0.04s | 1558 MB | Failed ✗ | 1.21x ✓ | Float16 unstable |

### Lazy Tensor Affinities (Memory Efficient)

| Affinity | Runtime (32-bit) | Memory (32-bit) | 16-bit Speedup | bf16 Speedup | Status |
|----------|-----------------|-----------------|----------------|--------------|---------|
| **PACMAP** | 0.02s | 1176 MB | 1.65x ✓ | 1.69x ✓ | Stable |
| **UMAP** | 0.04s | 1176 MB | 1.22x ✓ | 1.22x ✓ | Stable |
| **Entropic** | 0.61s | 1176 MB | 1.01x ✓ | 1.01x ✓ | Stable |

### Failed Affinities
- **SelfTuning**: Incorrect parameter name (should use `n_neighbors` instead of `perplexity`)
- **MAGIC**: Incorrect parameter name (should use `n_neighbors` instead of `knn`)

## Analysis

### Why Limited Mixed Precision Benefits?

1. **Memory Bandwidth Bound**: Affinity computations are primarily memory-bound operations
2. **Small Intermediate Tensors**: Most operations work with neighborhoods (30-50 points), not benefiting from tensor cores
3. **Overhead**: Type conversion overhead negates computational benefits
4. **Numerical Stability**: Several methods require float32 precision for convergence

### Output Type Comparison

- **Dense matrices** (10000×10000): High memory usage (~1.2-2GB) but direct access
- **Lazy tensors**: Memory efficient (~1.2GB) but require computation on access
- **Sparse matrices**: Not used by tested affinities (would be most efficient for large datasets)

## Recommendations

### For Speed Priority
Use **PACMAP**, **Student-t**, or **Gaussian** affinities with 32-bit precision

### For Memory Efficiency
Use **lazy tensor** affinities (PACMAP, UMAP, Entropic) which maintain ~1.2GB memory footprint

### For Stability
Avoid float16 with iterative methods (SymmetricEntropic, Sinkhorn). Use bfloat16 if mixed precision is needed.

### For Large-Scale Data
Consider sparse affinities (not benchmarked here) or batch processing approaches

## Conclusion

Traditional 32-bit precision remains the best choice for affinity computations in TorchDR due to:
- Minimal mixed precision benefits (< 2x speedup)
- No significant memory reduction
- Stability issues with float16
- Memory-bandwidth bottlenecks rather than compute bottlenecks

The fastest and most reliable affinities are simple dense computations (Student-t, Gaussian) or efficient lazy implementations (PACMAP, UMAP) running in float32.
