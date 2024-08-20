#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

__global__ void sieveKernel(bool* d_prime, int n, int chunkSize, int sq) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // Thread's unique index

    int chunkStart = idx * chunkSize + 2;
    int chunkEnd = min((idx + 1) * chunkSize + 1, n);
    //printf("chunks %d to %d", p, chunkEnd);
        
    if (chunkStart > chunkEnd || chunkStart < 2 || chunkStart > sq) return;

    //int p = chunkStart;

    // For each number up to sqrt(n), if it's prime, mark its multiples in this chunk
    for (int p = 2; p <= sq; ++p) {
        if (d_prime[p]) {
            // Start marking multiples of p within the chunk range
            for (int i = max(p * p, (chunkStart + p - 1) / p * p); i <= chunkEnd; i += p) {
                d_prime[i] = false; // Mark all multiples of p as not prime
            }
        }
    }
}

void sieveOfEratosthenes(int n) {
    bool* h_prime = (bool*)malloc((n + 1) * sizeof(bool)); // Host array
    bool* d_prime; // Device array

    // Initialize all entries as true (prime)
    for (int i = 0; i <= n; i++)
        h_prime[i] = true;

    h_prime[0] = h_prime[1] = false;

    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc((void**)&d_prime, (n + 1) * sizeof(bool));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the initialized prime array from the host to the device
    err = cudaMemcpy(d_prime, h_prime, (n + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int chunkSize = 1000; // Each thread processes 1000 numbers
    int numChunks = (n + chunkSize - 1) / chunkSize;
    int sq = sqrt(n);

    // Each block contains one thread
    int blockSize = 1;
    int numBlocks = numChunks;

    // Launch the kernel
    sieveKernel << <numBlocks, 128 >> > (d_prime, n, chunkSize, sq);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the result back to the host
    err = cudaMemcpy(h_prime, d_prime, (n + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (DeviceToHost) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print some prime numbers
     //for (int p = 2; p < 1000; p++) {
     //    if (h_prime[p]) {
     //        printf("%d ", p);
     //    }
     //}
    // printf("\n");

    // Free memory
    free(h_prime);
    cudaFree(d_prime);
}

int main() {
    int n = 1000000000; // Large number to test

    printf("Calculating the prime numbers smaller than or equal to %d:\n", n);

    // Start timing
    clock_t start = clock();

    sieveOfEratosthenes(n);

    // End timing
    clock_t end = clock();

    // Calculate the elapsed time in seconds
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nTime taken: %.2f seconds\n", time_spent);

    return 0;
}
