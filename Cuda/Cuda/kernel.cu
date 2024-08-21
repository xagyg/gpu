#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include<memory.h>


__global__ void sieveKernel(int* d_prime, int n, int sq) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // Thread's unique index

    if (idx > sq || idx < 2 || !d_prime[idx]) return;
    //printf("index %d, block, %d, thread %d, dim, %d\n", idx, blockIdx.x, threadIdx.x, blockDim.x);

    //if (d_prime[idx]) {
       // printf("marking %ds\n", idx);
        for (int i = idx * idx; i <= n; i += idx) {
            //atomicExch(&d_prime[i], 0); // Atomic operation to prevent overwrites
            //if (l_prime[i]) {
                d_prime[i] = 0;
                //printf("%d:%d ", idx, i);
            //}
        }
}


/**
__global__ void sieveKernel(int* d_prime, int n, int sq) {
    extern __shared__ int s_prime[];  // Shared memory

    int idx = blockDim.x * blockIdx.x + threadIdx.x;  // Global index
    int localIdx = threadIdx.x;  // Local index in shared memory

    // Load data into shared memory
    if (idx <= sq) {
        s_prime[localIdx] = d_prime[idx];
    }
    __syncthreads();

    // Sieve operation
    if (idx >= 2 && idx <= sq && s_prime[localIdx] == 1) {
        int start = idx * idx;
        for (int i = start; i <= n; i += idx) {
            //if (d_prime[i] == 1) {
                d_prime[i] = 0;
            //}
        }
    }
}
**/

void sieveOfEratosthenes(int n) {
    int* h_prime = (int*)malloc((n + 1) * sizeof(int)); // Host array
    int* d_prime; // Device array

    // Initialize all entries as true (prime)
    for (int i = 0; i <= n; i++)
        h_prime[i] = 1;

    h_prime[0] = h_prime[1] = 0;

    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc((void**)&d_prime, (n + 1) * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the initialized prime array from the host to the device
    err = cudaMemcpy(d_prime, h_prime, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int sq = sqrt(n);
    int threadsPerBlock = 1; // Adjust as needed
    int numBlocks = (sq + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemorySize = threadsPerBlock * sizeof(int);  // Shared memory size
    printf("num blocks %d\n", numBlocks);

    // Launch the kernel
    sieveKernel <<<numBlocks, threadsPerBlock >>> (d_prime, n, sq);
    //sieveKernel << <numBlocks, threadsPerBlock, sharedMemorySize >> > (d_prime, n, sq);

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
    err = cudaMemcpy(h_prime, d_prime, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (DeviceToHost) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int count = 0;
    for (int p = 2; p <= n; p++)
        if (h_prime[p]) {
            count++;
            //printf("%d ", p);
        }

    printf("\nNumber of primes: %d\n", count);

    // Free memory
    free(h_prime);
    cudaFree(d_prime);
}


int main() {
    int n = 100000000; // Large number to test

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
