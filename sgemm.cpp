#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.hpp>
#include <vector>

#define hipCheck(err) (hipCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for AMD hipBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12)
  {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL)
  {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  hipCheck(hipSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // HipDeviceInfo();

  // Declare the handle, create the handle, hipblasCreate will return a value of
  // type hipblasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  hipblasHandle_t handle;
  if (hipblasCreate(&handle))
  {
    std::cerr << "Create hipBLAS handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using hipEvent for gpu stream timing, hipEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  hipEvent_t beg, end;

  hipCheck(hipEventCreate(&beg));
  hipCheck(hipEventCreate(&end));

  // 4096 and 8192 yield similar results
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  hipCheck(hipMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  hipCheck(hipMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  hipCheck(hipMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  hipCheck(hipMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  hipCheck(hipMemcpy(dA, A, sizeof(float) * max_size * max_size,
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(dB, B, sizeof(float) * max_size * max_size,
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(dC, C, sizeof(float) * max_size * max_size,
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                     hipMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE)
  {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0)
    {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // hipBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      hipCheck(hipDeviceSynchronize());
      hipCheck(hipGetLastError()); // Check for async errors during kernel run
      hipCheck(hipMemcpy(C, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));
      hipCheck(hipMemcpy(C_ref, dC_ref, sizeof(float) * m * n, hipMemcpyDeviceToHost));

      if (!verify_matrix(C_ref, C, m * n))
      {
        std::cout
            << "Failed to pass the correctness verification against hipBLAS."
            << std::endl;
        if (m <= 128)
        {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    hipCheck(hipEventRecord(beg));
    for (int j = 0; j < repeat_times; j++)
    {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    hipCheck(hipEventRecord(end));
    hipCheck(hipEventSynchronize(beg));
    hipCheck(hipEventSynchronize(end));
    hipCheck(hipEventElapsedTime(&elapsed_time, beg, end));
    // elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) ms, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-6) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    hipCheck(hipMemcpy(dC, dC_ref, sizeof(float) * m * n,
                       hipMemcpyDeviceToDevice));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  hipCheck(hipFree(dA));
  hipCheck(hipFree(dB));
  hipCheck(hipFree(dC));
  hipCheck(hipFree(dC_ref));
  hipblasDestroy(handle);

  return 0;
};
