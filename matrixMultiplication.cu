#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cstdlib>

__global__ void matrixMultiplicationKernel(float* d_A, float* d_B, int m, int n){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row<m && col<n){
        int offset_A = row * m + col;
        int offset_B = col * n + row;
        d_A[offset_A] *= d_B[offset_B];
    }

}

void matrixMultiplication(float* h_A, float* h_B, int m, int n){
    auto start_assigning = std::chrono::high_resolution_clock::now();
    int size_d_A = m * n * sizeof(float);
    int size_d_B = m * n * sizeof(float);
    float *d_A, *d_B;

    cudaMalloc((void **)&d_A, size_d_A);
    cudaMemcpy(d_A, h_A, size_d_A, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size_d_B);
    cudaMemcpy(d_B, h_B, size_d_B, cudaMemcpyHostToDevice);

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);
    std::cout << "Cuda Assigning time taken : " << duration.count() << std::endl;

    auto start_cuda_execution = std::chrono::high_resolution_clock::now();
    dim3 nGrid(ceil(m/16), ceil(n/16), 1);
    dim3 nBlock(16, 16, 1);
    matrixMultiplicationKernel<<< nGrid, nBlock >>> (d_A, d_B, m, n);

    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);

    std::cout << "Cuda Execution time taken  on cuda : " << duration.count() << std::endl;
    
    cudaMemcpy(h_A, d_A, size_d_A, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B);

}


int main(){
    std::cout << "Parallel Run" << std::endl;
    auto start_assigning = std::chrono::high_resolution_clock::now();

    float mat_A[100][100], mat_B[100][100];
    int val = 0;

    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            val++;
            mat_A[i][j] = val;
            mat_B[j][i] = val;
        }
    }
    int rows = sizeof(mat_A)/sizeof(mat_A[0]);
    int cols = sizeof(mat_A[0])/sizeof(float);

    std::cout << rows << ", " << cols << std::endl;

    float* h_A = (float*)malloc(rows * cols * sizeof(float));
    float* h_B = (float*)malloc(rows * cols * sizeof(float));

    std::cout << "Memory allocated"<< std::endl;
    int offset = 0;

    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            offset = (rows*i + j);
            h_A[offset] = mat_A[i][j];
            h_B[offset] = mat_B[i][j];
        }
    }

    std::cout << "Memory filled with values" << std::endl;

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);

    std::cout << "Assigning time taken : " << duration.count() << std::endl;

    auto start_cuda_execution = std::chrono::high_resolution_clock::now();

    matrixMultiplication(&h_A[0], &h_B[0], rows, cols);
    
    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);

    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            offset = (rows*i + j);
            mat_A[i][j] = h_A[offset];
        }
    }

    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            std::cout << mat_A[i][j]<< ", ";
        }
        std::cout << std::endl;
    }

    return 1;
}