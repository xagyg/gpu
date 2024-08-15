
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

__global__ void sieveKernel(bool* d_prime, int n) {
    int p = blockDim.x * blockIdx.x + threadIdx.x + 2; // Each thread considers a number starting from 2
    if (p * p <= n && d_prime[p]) {
        for (int i = p * p; i <= n; i += p) {
            d_prime[i] = false; // Mark all multiples of p as not prime
        }
    }
}

void sieveOfEratosthenes(int n) {
    bool* h_prime = (bool*)malloc((n + 1) * sizeof(bool)); // Host array
    bool* d_prime; // Device array

    // Initialize all entries as true (prime)
    for (int i = 0; i <= n; i++)
        h_prime[i] = true;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_prime, (n + 1) * sizeof(bool));

    // Copy the initialized prime array from the host to the device
    cudaMemcpy(d_prime, h_prime, (n + 1) * sizeof(bool), cudaMemcpyHostToDevice);

    int blockSize = 256; // Define the block size
    int numBlocks = (n + blockSize - 1) / blockSize; // Calculate the number of blocks

    // Launch the kernel on the GPU
    sieveKernel <<<numBlocks, blockSize>>> (d_prime, n);

    // Copy the result back to the host
    cudaMemcpy(h_prime, d_prime, (n + 1) * sizeof(bool), cudaMemcpyDeviceToHost);

    // Print all prime numbers
    //for (int p = 2; p <= n; p++) {
    //    if (h_prime[p]) {
    //        printf("%d ", p);
    //    }
    //}

    // Free memory
    free(h_prime);
    cudaFree(d_prime);
}

int main() {
    int n = 1000000000; // Large number to test the CPU

    // Start timing
    clock_t start = clock();

    printf("Calculating the prime numbers smaller than or equal to %d:\n", n);
    sieveOfEratosthenes(n);

    // End timing
    clock_t end = clock();

    // Calculate the elapsed time in seconds
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nTime taken: %.2f seconds\n", time_spent);

    return 0;
}
