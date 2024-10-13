#include <iostream>
#include <ctime>
#include <chrono>

void print_result(int *c, long size) {
    for (int i = 0; i < size; i++) {
        std::cout << c[i] << ' ';
    }
    std::cout << std::endl;
}

void addl(int *a, int *b, int *c, long size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }    
}

__global__ void addcuda(int *a, int *b, int *c, long size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void validate(const int* a, const int* b, int size) {
    for(int i = 0; i < size; i++) {
        if(a[i] != b[i]) {
            std::cout << "Wrong result!" << std::endl;
            break;
        }
    }
}

int main() {
    const long size = 10000000;
    int *a = new int[size];
    int *b = new int[size];
    int *c = new int[size];
    int *dc = new int[size];
    srand(static_cast<unsigned>(time(0)));
    
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 11;
        b[i] = rand() % 11;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    addl(a, b, c, size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    // std::cout << "Loop result: ";
    // print_result(c, size);
    std::cout << "Time taken by CPU addition: " << cpu_duration.count() << " seconds\n";

    int *d_a, *d_b, *d_c;
    size_t bytes = size * sizeof(int);

    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    auto start_cuda = std::chrono::high_resolution_clock::now();
    addcuda<<<grid_size, block_size>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = end_cuda - start_cuda;
    std::cout << "Time taken by CUDA addition: " << cuda_duration.count() << " seconds\n";
    cudaMemcpy(dc, d_c, bytes, cudaMemcpyDeviceToHost);

    // std::cout << "CUDA result: ";
    // print_result(c, size);
    validate(dc, c, size);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
