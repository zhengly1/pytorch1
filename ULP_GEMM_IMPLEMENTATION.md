# ULP GEMM Implementation for PyTorch

This document describes the implementation of ULP (Unit in Last Place) perturbation functionality in PyTorch's CUDA GEMM operations.

## Overview

The implementation adds ULP perturbation to the left operand (matrix A) before each matrix multiplication operation in the `addmm_out_cuda_impl` function. This functionality is controlled by an environment variable and provides a fallback to standard cuBLAS operations.

## Implementation Details

### Files Modified/Added

1. **`aten/src/ATen/native/cuda/UlpGemm.h`** - New header file containing ULP perturbation functionality
2. **`aten/src/ATen/native/cuda/Blas.cpp`** - Modified to integrate ULP GEMM functionality

### Key Components

#### ULP Perturbation Functions

```cpp
__device__ inline float perturb_ulp_down(float value);
__device__ inline __half perturb_ulp_down(__half value);
```

These functions implement ULP perturbation by:
- For positive numbers: subtracting 1 from the mantissa bits
- For negative numbers: adding 1 to the mantissa bits (moves toward zero)
- Leaving zero values unchanged

#### ULP Perturbation Kernel

```cpp
template<typename T>
__global__ void apply_ulp_perturbation_kernel(T* data, int64_t size);
```

CUDA kernel that applies ULP perturbation to all elements of a matrix in parallel.

#### Main GEMM Function

```cpp
template<typename scalar_t>
bool gemm_with_ulp_perturbation(...);
```

Main function that:
1. Creates a temporary copy of matrix A
2. Applies ULP perturbation to the copy
3. Performs GEMM using cuBLAS with the perturbed matrix
4. Cleans up temporary memory

### Integration Points

The ULP GEMM functionality is integrated into `addmm_out_cuda_impl` with:

1. **Environment Variable Control**: `ENABLE_CUTLASS_ULP=1` enables the feature
2. **Conditional Execution**: ULP GEMM is used only when enabled and for supported types
3. **Fallback Mechanism**: Falls back to standard cuBLAS if ULP GEMM fails
4. **Type Support**: Currently supports float32 and half precision

## Usage

### Enabling ULP GEMM

```bash
export ENABLE_CUTLASS_ULP=1
python your_script.py
```

### Example Python Code

```python
import os
import torch

# Enable ULP GEMM
os.environ['ENABLE_CUTLASS_ULP'] = '1'

# Standard PyTorch operations will now use ULP GEMM
A = torch.randn(64, 64, device='cuda')
B = torch.randn(64, 64, device='cuda') 
C = torch.randn(64, 64, device='cuda')

# This will use ULP-perturbed GEMM
result = torch.addmm(C, A, B, alpha=2.0, beta=1.5)
```

## Technical Specifications

### ULP Perturbation Behavior

- **Direction**: Always perturbs DOWN (toward zero for negative numbers, toward smaller magnitude for positive numbers)
- **Magnitude**: Exactly one ULP (Unit in Last Place)
- **Precision**: Works with both float32 and half precision
- **Special Cases**: Zero values remain unchanged

### Performance Considerations

- **Memory Overhead**: Requires temporary allocation equal to size of matrix A
- **Compute Overhead**: Additional kernel launch for perturbation + memory copy
- **Fallback**: Automatic fallback to standard cuBLAS if ULP GEMM fails

### Supported Operations

- `torch.addmm()` and related matrix multiplication operations
- Both transposed and non-transposed matrices
- Float32 and half precision data types
- CUDA tensors only

## Testing and Verification

### Unit Tests

The implementation includes verification through:
1. **ULP Perturbation Test**: Verifies correct bit-level perturbation
2. **GEMM Simulation**: Demonstrates effect on matrix multiplication
3. **Integration Test**: Tests with PyTorch operations

### Expected Behavior

- **Small Differences**: Results should differ from standard GEMM by small amounts (typically 1e-7 to 1e-6 range)
- **Deterministic**: Same inputs produce same perturbed outputs
- **Robust**: Fallback ensures operations always complete successfully

## Limitations and Considerations

1. **CUDA Only**: Implementation is specific to CUDA tensors
2. **Memory Usage**: Requires additional GPU memory for temporary matrix copy
3. **Performance Impact**: Slight overhead due to perturbation kernel and memory operations
4. **Mixed Precision**: Currently limited support for mixed precision operations

## Future Enhancements

Potential improvements could include:
1. **In-place Perturbation**: Modify original matrix to avoid memory copies
2. **CUTLASS Integration**: True CUTLASS kernel with built-in perturbation
3. **Batch Support**: Optimized handling of batch operations
4. **CPU Implementation**: Extend to CPU operations

## Debugging and Troubleshooting

### Environment Variables

- `ENABLE_CUTLASS_ULP=1`: Enable ULP GEMM functionality
- `ENABLE_CUTLASS_ULP=0`: Disable and use standard cuBLAS (default)

### Common Issues

1. **CUDA Out of Memory**: ULP GEMM requires additional memory for matrix copy
2. **Type Mismatch**: Ensure tensors are supported types (float32/half)
3. **Device Placement**: Tensors must be on CUDA device

### Verification

Use the provided verification script to test ULP perturbation:

```bash
python verify_ulp_implementation.py
```

This will demonstrate the ULP perturbation effects and validate the implementation logic.