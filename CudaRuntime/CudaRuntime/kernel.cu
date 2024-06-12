
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 2048

using namespace std;


__global__ void mul_matrix(double* matrix_1, double* matrix_2, double* matrix_mul) {
    double value;
    int start_row = blockIdx.x, count_blocks = gridDim.x;
    int start_col = threadIdx.x, cout_thread = blockDim.x;

    for (int i = start_row; i < N; i += count_blocks)
        for (int j = start_col; j < N; j += cout_thread) {
            value = 0;
            for (int k = 0; k < N; ++k)
                value += matrix_1[i * N + k] * matrix_2[k * N + j];
            matrix_mul[i * N + j] = value;
        }
}

int main() {
    double* matrix_1, * matrix_2, * matrix_mul;
    int size = N * N * sizeof(double);
    cudaEvent_t start, stop;
    float gpu_time;

    cudaMallocManaged(&matrix_1, size);
    cudaMallocManaged(&matrix_2, size);
    cudaMallocManaged(&matrix_mul, size);

    for (int i = 0; i < N * N; ++i)
        matrix_1[i] = matrix_2[i] = 2;

    int _blocks = 32, _threads = 1024;
    dim3 threads(_threads);
    dim3 blocks(_blocks);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mul_matrix << <blocks, threads >> > (matrix_1, matrix_2, matrix_mul);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Cout blocks = %i , count threads = %i , time = %f", _blocks, _threads, gpu_time);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(matrix_1); cudaFree(matrix_2); cudaFree(matrix_mul);
}