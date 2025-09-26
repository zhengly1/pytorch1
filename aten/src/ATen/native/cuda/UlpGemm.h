#pragma once

#include <c10/util/Exception.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDABlas.h>

namespace at::native {

// ULP perturbation operation for floats
__device__ inline float perturb_ulp_down(float value) {
  if (value == 0.0f) return value;
  
  // Get the bit representation
  union { float f; uint32_t i; } u = {value};
  
  // If positive, subtract 1 from the mantissa (perturb down)
  // If negative, add 1 to the mantissa (perturb towards zero, which is down in magnitude)
  if ((u.i & 0x80000000U) == 0) {
    // Positive number - subtract 1
    if (u.i > 0) u.i--;
  } else {
    // Negative number - add 1 (moves towards zero)
    u.i++;
  }
  
  return u.f;
}

// ULP perturbation operation for half precision
__device__ inline __half perturb_ulp_down(__half value) {
  if (__heq(value, __float2half(0.0f))) return value;
  
  // Get the bit representation
  uint16_t bits = __half_as_ushort(value);
  
  // If positive, subtract 1 from the mantissa
  // If negative, add 1 to the mantissa (moves towards zero)
  if ((bits & 0x8000U) == 0) {
    // Positive number - subtract 1
    if (bits > 0) bits--;
  } else {
    // Negative number - add 1
    bits++;
  }
  
  return __ushort_as_half(bits);
}

// Simple kernel to apply ULP perturbation to a matrix in-place
template<typename T>
__global__ void apply_ulp_perturbation_kernel(T* data, int64_t size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = perturb_ulp_down(data[idx]);
  }
}

// Function to apply ULP perturbation to a tensor
template<typename T>
void apply_ulp_perturbation(T* data, int64_t size, cudaStream_t stream = 0) {
  const int block_size = 256;
  const int grid_size = (size + block_size - 1) / block_size;
  apply_ulp_perturbation_kernel<<<grid_size, block_size, 0, stream>>>(data, size);
}

// Enhanced approach: use cuBLAS GEMM with pre-perturbation of input A
template<typename scalar_t>
bool gemm_with_ulp_perturbation(
  bool transpose_a,
  bool transpose_b,
  int64_t m,
  int64_t n,
  int64_t k,
  at::opmath_type<scalar_t> alpha,
  const scalar_t* a_ptr,
  int64_t lda,
  const scalar_t* b_ptr,
  int64_t ldb,
  at::opmath_type<scalar_t> beta,
  scalar_t* c_ptr,
  int64_t ldc
) {
  try {
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create a temporary copy of matrix A and apply ULP perturbation
    size_t a_size = m * k;
    scalar_t* a_perturbed;
    cudaError_t cuda_status = cudaMalloc(&a_perturbed, a_size * sizeof(scalar_t));
    if (cuda_status != cudaSuccess) return false;
    
    // Copy A to temporary buffer
    cudaMemcpyAsync(a_perturbed, a_ptr, a_size * sizeof(scalar_t), cudaMemcpyDeviceToDevice, stream);
    
    // Apply ULP perturbation to the copy
    apply_ulp_perturbation(a_perturbed, a_size, stream);
    
    // Wait for perturbation to complete
    cudaStreamSynchronize(stream);
    
    // Use cuBLAS GEMM with the perturbed A matrix
    char transa_char = transpose_a ? 't' : 'n';
    char transb_char = transpose_b ? 't' : 'n';
    
    at::cuda::blas::gemm<scalar_t>(
        transa_char,
        transb_char,
        m,
        n,
        k,
        alpha,
        a_perturbed,
        lda,
        b_ptr,
        ldb,
        beta,
        c_ptr,
        ldc);
    
    // Clean up
    cudaFree(a_perturbed);
    
    return true;
  }
  catch (...) {
    return false;
  }
}

} // namespace at::native