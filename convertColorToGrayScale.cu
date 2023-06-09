#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cstdlib>

__global__ void convertColorToGrayScaleKernel(float* A, float* B, int m, int n){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < m && row < n){
        int grayOffset = row*m + col;
        float r = A[grayOffset * 3 + 0];
        float g = A[grayOffset * 3 + 1];
        float b = A[grayOffset * 3 + 2];

        B[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void convertColorToGrayScale(float* h_A, float* h_B, int m, int n){
    auto start_assigning = std::chrono::high_resolution_clock::now();
    
    int size_d_A = 3 * sizeof(float) * m * n;
    int size_d_B = sizeof(float) * m * n;
    float *d_A, *d_B;
    
    cudaMalloc((void **)&d_A, size_d_A);
    cudaMemcpy(d_A, h_A, size_d_A, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size_d_B);

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);
    std::cout << "Cuda Assigning time taken : " << duration.count() << std::endl;

    auto start_cuda_execution = std::chrono::high_resolution_clock::now();
    dim3 nGrid(ceil(m/16), ceil(n/16), 1);
    dim3 nBlocks(16, 16, 1);
    convertColorToGrayScaleKernel<<<nGrid, nBlocks>>> (d_A, d_B, m, n);

    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);

    std::cout << "Cuda Execution time taken  on cuda : " << duration.count() << std::endl;
    
    cudaMemcpy(h_B, d_B, size_d_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B);
}

int main(){
    std::cout << "Parallel Run" << std::endl;
    auto start_assigning = std::chrono::high_resolution_clock::now();

    int m = 1024 , n = 1024;
    int length = m*n;
    float* h_A = (float*)malloc(3 * length * sizeof(float));
    float* h_B = (float*)malloc(length * sizeof(float));

    for(int i=0; i<length; i++){
        h_A[i*3 + 0] = std::rand()%255;
        h_A[i*3 + 1] = std::rand()%255;
        h_A[i*3 + 2] = std::rand()%255;

        h_B[i] = 0.0f; 
    }

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);

    std::cout << "Assigning time taken : " << duration.count() << std::endl;
    
    auto start_cuda_execution = std::chrono::high_resolution_clock::now();

    convertColorToGrayScale(&h_A[0], &h_B[0], m, n);
    
    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            std::cout << h_B[j + i] << " , ";
        }
        std::cout<< std::endl;
    }
    std::cout<< std::endl;

    std::cout << "Parallel Run Completed" << std::endl;
    
}