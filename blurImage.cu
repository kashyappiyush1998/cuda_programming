#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

__device__ void blurImageChannelKernel() {

}

__global__ void blurImageKernel(float* a, float* b, int m, int n){
    int row = blockDim.x * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    // int offset = 3*(row * m + col);

    if(row < m && col < n){
        int offset = 3*(row * m + col);
        int blockWidth = 3;
        for(int k=0 ; k < 3; k++){
            int nPixels = 0;
            float sum = 0;
            for(int i=-blockWidth ;i < blockWidth; i++){
                for(int j = -blockWidth; j < blockWidth; j++){
                    sum += a[offset + i*3 + j*m*3];
                    nPixels++;
                }
            }
            b[offset] = sum/nPixels;
            offset += 1;
        }
        // b[offset] = (a[offset] + a[offset + 3] + a[offset - 3] + a[offset - m*3] + a[offset - m*3 + 3] + a[offset - m*3 - 3] + a[offset + m*3 + 3] + a[offset + m*3 - 3])/9.0f;
        // offset += 1;
        // b[offset] = (a[offset] + a[offset + 3] + a[offset - 3] + a[offset - m*3] + a[offset - m*3 + 3] + a[offset - m*3 - 3] + a[offset + m*3 + 3] + a[offset + m*3 - 3])/9.0f;
        // offset += 1;
        // b[offset] = (a[offset] + a[offset + 3] + a[offset - 3] + a[offset - m*3] + a[offset - m*3 + 3] + a[offset - m*3 - 3] + a[offset + m*3 + 3] + a[offset + m*3 - 3])/9.0f;
    }
}

void blurImage(float* h_A, float* h_B, int m, int n){
    auto start_assigning = std::chrono::high_resolution_clock::now();
    
    int size_d_A = 3 * sizeof(float) * m * n;
    int size_d_B = 3 * sizeof(float) * m * n;
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
    blurImageKernel<<<nGrid, nBlocks>>> (d_A, d_B, m, n);

    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);

    std::cout << "Cuda Execution time taken  on cuda : " << duration.count() << std::endl;
    
    cudaMemcpy(h_B, d_B, size_d_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B);
}

int main(){
    std::cout << "Parallel Run" << std::endl;
    auto start_assigning = std::chrono::high_resolution_clock::now();

    std::string inpFilePath = "/data/rishubh/piyush/cuda_programming/images/bike_500_11zon.png";
    cv::Mat img = cv::imread(inpFilePath, cv::IMREAD_COLOR);
    cv::Mat img2;
    if(img.empty())
    {
        std::cout << "Could not read the image: " << inpFilePath << std::endl;
        return 1;
    }
    cv::resize(img, img, cv::Size(1024, 1024));
    img.convertTo(img, CV_32FC3);

    std::cout << img.rows << ", " << img.cols << std::endl;

    int m = 1024 , n = 1024;
    int length = m*n;
    float* h_A = (float*)malloc(3 * length * sizeof(float));
    float* h_B = (float*)malloc(3 * length * sizeof(float));

    std::cout << "Memory allocated"<< std::endl;
    int offset = 0;

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            offset = (i*m+j)*3;
            // std::cout << i << " " << j << " " << offset<<std::endl;
            cv::Vec3f channels = img.at<cv::Vec3f>(i,j);
            h_A[offset + 0] = channels[0];  h_B[offset + 0] = 0.0f;
            h_A[offset + 1] = channels[1];  h_B[offset + 1] = 0.0f;
            h_A[offset + 2] = channels[2];  h_B[offset + 2] = 0.0f;

        }
    }
    std::cout << "Memory filled with values" << std::endl;

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);

    std::cout << "Assigning time taken : " << duration.count() << std::endl;
    
    auto start_cuda_execution = std::chrono::high_resolution_clock::now();

    blurImage(&h_A[0], &h_B[0], m, n);
    
    auto stop_cuda_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cuda_execution - start_cuda_execution);
    
    cv::Mat out = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC3);

    cv::Vec3f assignPixels;
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            offset = 3*(i*m+j);
            assignPixels[0] = h_B[offset + 0];
            assignPixels[1] = h_B[offset + 1];
            assignPixels[2] = h_B[offset + 2];
            out.at<cv::Vec3f>(i,j) = assignPixels;
        }
    }

    std::string outFilePath = "/data/rishubh/piyush/cuda_programming/output_images/bike_500_blur_block_7.png";
    cv::imwrite(outFilePath, out);

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            std::cout << h_B[j + i] << " , ";
        }
        std::cout<< std::endl;
    }
    std::cout<< std::endl;

    std::cout << "Parallel Run Completed" << std::endl;
    
}