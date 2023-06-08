#include <iostream>
#include <chrono>

void vecAdd(float* h_A, float* h_B, float* h_C, int n){
    for(int i=0; i<10000; i++){
        h_C[i] = h_A[i]+ h_B[i];
    }
}

int main(){
    std::cout << "Sequential Run" << std::endl;

    auto start_assigning = std::chrono::high_resolution_clock::now();

    float h_A[10000], h_B[10000], h_C[10000];
    for(int i=0; i<10000; i++){
        h_A[i] = i; 
        h_B[i] = i;
        h_C[i] = 0;
    }

    auto stop_assigning = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_assigning - start_assigning);

    std::cout << "Assigning time taken : " << duration.count() << std::endl;
    
    auto start_execution = std::chrono::high_resolution_clock::now();

    vecAdd(&h_A[0], &h_B[0], &h_C[0], 10000);
    
    auto stop_execution = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_execution - start_execution);

    std::cout << "Execution time taken : " << duration.count() << std::endl;


    for(int i=0; i<10; i++){
        std::cout << h_C[i] << " , ";
    }
    std::cout<< std::endl;

    std::cout << "Sequential Run Completed" << std::endl;
    

}