#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "kernels.hpp"
#include "runner.hpp"
#include "utils.hpp"

void hipCheck(hipError_t error, const char *file, int line)
{
  if (error != hipSuccess)
  {
    printf("[HIP ERROR] at file %s:%d:\n%s\n", file, line,
           hipGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void HipDeviceInfo()
{
  int deviceId;

  hipError_t err;
  err = hipGetDevice(&deviceId);
  if (err != hipSuccess)
  {
    std::cerr << "Error: hipGetDevice failed with error code " << hipGetErrorString(err) << std::endl;
    return;
  }

  hipDeviceProp_t props{};
  err = hipGetDeviceProperties(&props, deviceId);

  if (err != hipSuccess)
  {
    std::cerr << "Error: hipGetDeviceProperties failed with error code " << hipGetErrorString(err) << std::endl;
    return;
  }

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N)
{
  struct timeval time
  {
  };
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++)
  {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs)
{
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++)
  {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0)
    {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N)
{
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++)
  {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01)
    {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

void runHipBlasFP32(hipblasHandle_t handle, int M, int N, int K, float alpha,
                    float *A, float *B, float beta, float *C)
{
  // hipBLAS uses column-major order.
  // So we change the order of our row-major A & B, since (B^T*A^T)^T = (A*B)
  hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &alpha, B, HIPBLAS_R_32F,
                N, A, HIPBLAS_R_32F, K, &beta, C, HIPBLAS_R_32F, N, HIPBLAS_COMPUTE_32F,
                HIPBLAS_GEMM_DEFAULT);
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C)
{
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  hipLaunchKernelGGL(sgemm_naive, gridDim, blockDim, 0, 0, M, N, K, alpha, A, B, beta, C);
}

// void run_sgemm_naive_occupancy(int M, int N, int K, float alpha, float *A, float *B,
//                          float beta, float *C)
// {
//     int minGridSize, blockSize;

//     // Calculate the optimal block size and minimum grid size
//     hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
//                                       (void *)sgemm_naive, 0, 0);

//     printf("Optimal block size (total threads): %d\n", blockSize);

//     // Convert blockSize (total threads per block) to 2D block dimensions
//     // Here, we assume a square block size for simplicity
//     int blockDimX = sqrt(blockSize); // Set blockDim.x as the square root of blockSize
//     int blockDimY = blockSize / blockDimX; // Set blockDim.y to make up the rest

//     printf("Optimal block dimensions: %d x %d\n", blockDimX, blockDimY);

//     // Calculate grid size based on the new block dimensions
//     int gridSizeX = CEIL_DIV(M, blockDimX);
//     int gridSizeY = CEIL_DIV(N, blockDimY);

//     dim3 gridDim(gridSizeX, gridSizeY);
//     dim3 blockDim(blockDimX, blockDimY);

//     // Launch kernel with optimized block and grid dimensions
//     hipLaunchKernelGGL(sgemm_naive, gridDim, blockDim, 0, 0, M, N, K, alpha, A, B, beta, C);
// }


void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C)
{
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C)
{
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);

  /// FIXME: cudaSharedmemCarveoutMaxShared equivalence is not defined in HIP

  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  // cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
  //                      cudaFuncAttributePreferredSharedMemoryCarveout,
  //                      cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128)
  {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
  else
  {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C)
{
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128)
  {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
  else
  {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C)
{
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128)
  {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
  else
  {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C)
{
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128)
  {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
  else
  {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

// A100
// const uint K9_BK = 16;
// const uint K9_TM = 4;
// const uint K9_TN = 4;
// const uint K9_BM = 64;
// const uint K9_BN = 64;
// A6000
// const uint K9_BK = 16;
// const uint K9_TM = 8;
// const uint K9_TN = 8;
// const uint K9_BM = 128;
// const uint K9_BN = 128;

#ifndef K9_NUM_THREADS
#define K9_NUM_THREADS 256
#endif

#ifndef K9_BK
#define K9_BK 16
#endif

#ifndef K9_TM
#define K9_TM 8
#endif

#ifndef K9_TN
#define K9_TN 8
#endif

#ifndef K9_BM
#define K9_BM 128
#endif

#ifndef K9_BN
#define K9_BN 128
#endif

void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C)
{
  dim3 blockDim(K9_NUM_THREADS);

  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

  // print current configuration in one line
  // printf("K9_NUM_THREADS=%d K9_BN=%d K9_BM=%d K9_BK=%d K9_TN=%d K9_TM=%d\n",
  //        K9_NUM_THREADS, K9_BN, K9_BM, K9_BK, K9_TN,
  //        K9_TM);

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN, K9_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// Settings for A100
// const uint K10_NUM_THREADS = 128;
// const uint K10_BN = 64;
// const uint K10_BM = 64;
// const uint K10_BK = 8;
// const uint K10_WN = 32;
// const uint K10_WM = 32;
// const uint K10_WNITER = 2;
// const uint K10_TN = 4;
// const uint K10_TM = 4;

// Settings for A6000
// const uint K10_NUM_THREADS = 128;
// const uint K10_BN = 64;
// const uint K10_BM = 64;
// const uint K10_BK = 8;
// const uint K10_WN = 32;
// const uint K10_WM = 32;
// const uint K10_WNITER = 2;
// const uint K10_TN = 4;
// const uint K10_TM = 4;
#ifndef K10_NUM_THREADS
#define K10_NUM_THREADS 128
#endif

#ifndef K10_BN
#define K10_BN 128
#endif

#ifndef K10_BM
#define K10_BM 128
#endif

#ifndef K10_BK
#define K10_BK 16
#endif

#ifndef K10_WM
#define K10_WM 64
#endif

#ifndef K10_WN
#define K10_WN 64
#endif

#ifndef K10_WNITER
#define K10_WNITER 4
#endif

#ifndef K10_TN
#define K10_TN 4
#endif

#ifndef K10_TM
#define K10_TM 8
#endif

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C)
{
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  // printf("K10_NUM_THREADS=%d K10_BN=%d K10_BM=%d K10_BK=%d K10_WM=%d K10_WN=%d "
  //        "K10_WNITER=%d K10_TM=%d K10_TN=%d\n",
  //        K10_NUM_THREADS, K10_BN, K10_BM, K10_BK, K10_WM, K10_WN, K10_WNITER,
  //        K10_TM, K10_TN);

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, hipblasHandle_t handle)
{
  switch (kernel_num)
  {
  case 0:
    runHipBlasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  // case 11:
  //   run_sgemm_naive_occupancy(M, N, K, alpha, A, B, beta, C);
  //   break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}