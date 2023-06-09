#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <chrono>
#include <iostream>

// Compute vector sum C = A + B
// Each thread performs pair wise operations
__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<n){
        C[i] = A[i] + B[i];
    }
}

void vecAddCuda(float* h_A, float* h_B, float* h_C, int n){

    auto start_assigning = std::chrono::high_resolution_clock::now();

    int size = sizeof(float) * n;
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, size);

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);
    std::cout << "Cuda Assigning time taken : " << duration.count() << std::endl;

    auto start_cuda_execution = std::chrono::high_resolution_clock::now();

    vecAddKernel<<<ceil(n/256.0), 256>>> (d_A, d_B, d_C, n);

    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);

    std::cout << "Cuda Execution time taken  on cuda : " << duration.count() << std::endl;

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory for d_A, d_B, d_C
    cudaFree(d_A); cudaFree(d_B), cudaFree(d_C);
}

int main(){
    std::cout << "Parallel Run" << std::endl;

    auto start_assigning = std::chrono::high_resolution_clock::now();

    float h_A[100000], h_B[100000], h_C[100000];
    for(int i=0; i<100000; i++){
        h_A[i] = i; 
        h_B[i] = i;
        h_C[i] = 0;
    }

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);

    std::cout << "Assigning time taken : " << duration.count() << std::endl;

    auto start_cuda_execution = std::chrono::high_resolution_clock::now();

    vecAddCuda(&h_A[0], &h_B[0], &h_C[0], 100000);
    
    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);

    std::cout << "Total Execution time taken  on cuda : " << duration.count() << std::endl;


    for(int i=0; i<10; i++){
        std::cout << h_C[i] << " , ";
    }
    std::cout<< std::endl;

    std::cout << "Parallel Run Completed" << std::endl;
}