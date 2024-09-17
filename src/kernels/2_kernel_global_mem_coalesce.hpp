#ifndef KERNELS_2_KERNEL_GLOBAL_MEM_COALESCE_HPP
#define KERNELS_2_KERNEL_GLOBAL_MEM_COALESCE_HPP

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include "../utils.hpp"

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C)
{
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

#endif // KERNELS_2_KERNEL_GLOBAL_MEM_COALESCE_HPP