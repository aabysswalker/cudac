#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <chrono>

void print_array(const int* a, int size) {
    for(int i = 0; i < size; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void sort_cpu(int* a, int size) {
    bool swapped;
  
    for (int i = 0; i < size; i++) {
        swapped = false;
        for (int j = 0; j < size - 1; j++) {
            if (a[j] > a[j + 1]) {
                std::swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
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
    int size = 30000;
    int a[size];
    int v[size];
    srand(static_cast<unsigned>(time(0)));
    
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 100;
        v[i] = a[i];
    }


    auto start_gpu = std::chrono::high_resolution_clock::now();
    thrust::device_vector<int> d_a(a, a + size);
    thrust::sort(d_a.begin(), d_a.end());
    thrust::copy(d_a.begin(), d_a.end(), a);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "Time taken device: " << gpu_duration.count() << " seconds\n";
    // print_array(a, size);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    sort_cpu(v, size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "Time taken host: " << cpu_duration.count() << " seconds\n";
    // print_array(v, size);

    validate(a,v,size);

    return 0;
}
