#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>

struct Matrix {
    int r;
    int c;
    int** data;

    Matrix(int row, int columns) : r(row), c(columns) {
        data = new int*[r];
        for (int i = 0; i < r; i++) {
            data[i] = new int[c];
        }
    }

    ~Matrix() {
        for (int i = 0; i < r; i++) {
            delete[] data[i];
        }
        delete[] data;
    }

    int* align_device() const {
        int* arr = new int[r * c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                arr[i * c + j] = data[i][j];
            }
        }
        return arr;
    }

    void align_host(int* arr) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                data[i][j] = arr[i * c + j];
            }
        }
    }

    void fill_random() {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                data[i][j] = rand() % 11;
            }
        }
    }

    void print() {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                std::cout << data[i][j] << ' ';
            }
            std::cout << std::endl;
        }
    }
};

void multiply(Matrix& m1, Matrix& m2, Matrix& result) {
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m1.r; i++) {
        for (int j = 0; j < m2.c; j++) {
            result.data[i][j] = 0;
            for (int k = 0; k < m1.c; k++) {
                result.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "Time taken host: " << cpu_duration.count() << " seconds\n";
}

#define TILE_WIDTH 16

__global__ void matrix_kernel(int* m1, int* m2, int* result, int m1Rows, int m1Cols, int m2Cols) {
    __shared__ int shared_m1[TILE_WIDTH][TILE_WIDTH];
    __shared__ int shared_m2[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int sum = 0;

    for (int t = 0; t < (m1Cols + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {

        if (row < m1Rows && t * TILE_WIDTH + threadIdx.x < m1Cols) {
            shared_m1[threadIdx.y][threadIdx.x] = m1[row * m1Cols + t * TILE_WIDTH + threadIdx.x];
        } else {
            shared_m1[threadIdx.y][threadIdx.x] = 0;
        }

        if (t * TILE_WIDTH + threadIdx.y < m1Cols && col < m2Cols) {
            shared_m2[threadIdx.y][threadIdx.x] = m2[(t * TILE_WIDTH + threadIdx.y) * m2Cols + col];
        } else {
            shared_m2[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_m1[threadIdx.y][k] * shared_m2[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m1Rows && col < m2Cols) {
        result[row * m2Cols + col] = sum;
    }
}

void multiplyc(Matrix& m1, Matrix& m2, Matrix& result) {
    int* d_m1, *d_m2, *d_result;
    int size_m1 = m1.r * m1.c * sizeof(int);
    int size_m2 = m2.r * m2.c * sizeof(int);
    int size_result = result.r * result.c * sizeof(int);

    cudaMalloc((void**)&d_m1, size_m1);
    cudaMalloc((void**)&d_m2, size_m2);
    cudaMalloc((void**)&d_result, size_result);

    int* m1_1d = m1.align_device();
    int* m2_1d = m2.align_device();

    cudaMemcpy(d_m1, m1_1d, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2_1d, size_m2, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((result.c + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (result.r + threadsPerBlock.y - 1) / threadsPerBlock.y);


    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1, d_m2, d_result, m1.r, m1.c, m2.c);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "Time taken device: " << gpu_duration.count() << " seconds\n";
    

    int* result_1d = new int[result.r * result.c];
    cudaMemcpy(result_1d, d_result, size_result, cudaMemcpyDeviceToHost);

    result.align_host(result_1d);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    delete[] m1_1d;
    delete[] m2_1d;
    delete[] result_1d;
}

bool validate(Matrix& m1, Matrix& m2) {
    for (int i = 0; i < m1.r; i++) {
        for (int j = 0; j < m1.c; j++) {
            if (m1.data[i][j] != m2.data[i][j]) {
                std::cout << "Wrong result!" << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    srand(static_cast<unsigned>(time(0))); 

    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    Matrix result(m1.r, m2.c);
    Matrix resultc(m1.r, m2.c);
    m1.fill_random();
    m2.fill_random();

    // std::cout << "Matrix A: " << std::endl;
    // m1.print();
    // std::cout << "Matrix B: " << std::endl;
    // m2.print();

    multiply(m1, m2, result);
    multiplyc(m1, m2, resultc);

    // resultc.print();

    if(validate(result, resultc)) {
        // std::cout << "Matrix A * B: " << std::endl;
        // result.print();
        std::cout << "Valid result" << std::endl;
    } else {
        return 0;
    }


    return 0;
}
