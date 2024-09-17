#include <cstdio>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

/*
 * A stand-alone script to invoke & benchmark standard hipBLAS SGEMM performance
 */

// Define the hipCheck macro for error checking
#define hipCheck(error) (hipCheckFn(error, __FILE__, __LINE__))

void hipCheckFn(hipError_t error, const char *file, int line) {
  if (error != hipSuccess) {
    printf("[HIP ERROR] at file %s:%d:\n%s\n", file, line, hipGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  int m = 2;
  int k = 3;
  int n = 4;
  int print = 1;
  hipblasStatus_t stat;     // hipBLAS functions status
  hipblasHandle_t handle;   // hipBLAS context

  int i, j;

  float *a, *b, *c;

  // malloc for a, b, c...
  a = (float *)malloc(m * k * sizeof(float));
  b = (float *)malloc(k * n * sizeof(float));
  c = (float *)malloc(m * n * sizeof(float));

  int ind = 11;
  for (j = 0; j < m * k; j++) {
    a[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < k * n; j++) {
    b[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < m * n; j++) {
    c[j] = (float)ind++;
  }

  // DEVICE
  float *d_a, *d_b, *d_c;

  // hipMalloc for d_a, d_b, d_c...
  hipCheck(hipMalloc((void **)&d_a, m * k * sizeof(float)));
  hipCheck(hipMalloc((void **)&d_b, k * n * sizeof(float)));
  hipCheck(hipMalloc((void **)&d_c, m * n * sizeof(float)));

  stat = hipblasCreate(&handle); // initialize hipBLAS context
  if (stat != HIPBLAS_STATUS_SUCCESS) {
    printf("hipBLAS initialization failed\n");
    return EXIT_FAILURE;
  }

  hipCheck(hipMemcpy(d_a, a, m * k * sizeof(float), hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_b, b, k * n * sizeof(float), hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_c, c, m * n * sizeof(float), hipMemcpyHostToDevice));

  float alpha = 1.0f;
  float beta = 0.5f;

  if (print == 1) {
    printf("alpha = %4.0f, beta = %4.0f\n", alpha, beta);
    printf("A = (mxk: %d x %d)\n", m, k);
    for (i = 0; i < m; i++) {
      for (j = 0; j < k; j++) {
        printf("%4.1f ", a[i * m + j]);
      }
      printf("\n");
    }
    printf("B = (kxn: %d x %d)\n", k, n);
    for (i = 0; i < k; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", b[i * n + j]);
      }
      printf("\n");
    }
    printf("C = (mxn: %d x %d)\n", m, n);
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  stat = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k, &alpha, d_b, n,
                      d_a, k, &beta, d_c, n);

  if (stat != HIPBLAS_STATUS_SUCCESS) {
    printf("hipBLAS SGEMM failed\n");
    return EXIT_FAILURE;
  }

  hipCheck(hipMemcpy(c, d_c, m * n * sizeof(float), hipMemcpyDeviceToHost));

  if (print == 1) {
    printf("\nC after SGEMM = \n");
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  hipCheck(hipFree(d_a));
  hipCheck(hipFree(d_b));
  hipCheck(hipFree(d_c));
  hipblasDestroy(handle); // destroy hipBLAS context
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
