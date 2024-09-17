#pragma once
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>


void hipCheck(hipError_t error, const char *file,
              int line); // HIP error check
void HipDeviceInfo();    // print HIP information

void randomize_matrix(float *mat, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, hipblasHandle_t handle);
