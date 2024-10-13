#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

void print_array(const float* a, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << std::fixed << std::setprecision(1) << a[i] << std::endl;
    }
}

void add_array(const float* a, int size, float& res) {
    for (int i = 0; i < size; i++) {
        res += a[i];
    }
}

__global__ void add_array_cu(const float* a, int size, float* res) {
    extern __shared__ float shared_data[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        shared_data[threadIdx.x] = a[index];
    } else {
        shared_data[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, shared_data[0]);
    }
}

void reduce_array(const float* h_a, int size, float& res) {
    float* d_a;
    float* d_res;
    float h_res = 0.0f;

    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    add_array_cu<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_a, size, d_res);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "Time taken device: " << gpu_duration.count() << " seconds\n";
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    res = h_res;

    cudaFree(d_a);
    cudaFree(d_res);
}

int main() {
    int size = 100000;
    float arr[size];
    float lres = 0, cres = 0;

    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < size; i++) {
        arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_array(arr, size, lres);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "Time taken host: " << cpu_duration.count() << " seconds\n";
    std::cout << "Loop result: " << std::fixed << std::setprecision(6) << lres << std::endl;
    
    reduce_array(arr, size, cres);
    std::cout << "CUDA result: " << std::fixed << std::setprecision(6) << cres << std::endl;

    return 0;
}
