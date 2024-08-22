#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <windows.h>
#include <math.h>

#define NUM_THREADS 6  // Number of threads

typedef struct {
    bool* prime;
    int start;
    int end;
    int p;
} ThreadData;

DWORD WINAPI markMultiples(LPVOID arg) {
    ThreadData* data = (ThreadData*)arg;
    bool* prime = data->prime;
    int start = data->start;
    int end = data->end;
    int p = data->p;

    for (int i = start; i <= end; i += p) {
        prime[i] = false;
    }

    return 0;
}

void sieveOfEratosthenes(int n) {
    bool* prime = (bool*)malloc((n + 1) * sizeof(bool));
    for (int i = 0; i <= n; i++) {
        prime[i] = true;
    }

    int sq = sqrt(n);

    for (int p = 2; p <= sq; p++) {
        if (prime[p] == true) {
            HANDLE threads[NUM_THREADS];
            ThreadData threadData[NUM_THREADS];

            // Adjust the block size calculation to avoid overlap
            int blockSize = (n / NUM_THREADS) + 1;

            for (int t = 0; t < NUM_THREADS; t++) {
                threadData[t].prime = prime;
                threadData[t].p = p;
                threadData[t].start = p * p + t * blockSize;

                // Ensure that start is a multiple of p
                if (threadData[t].start % p != 0) {
                    threadData[t].start += p - (threadData[t].start % p);
                }

                threadData[t].end = (t == NUM_THREADS - 1) ? n : (threadData[t].start + blockSize - 1);

                // Ensure that the end does not exceed n
                if (threadData[t].end > n) {
                    threadData[t].end = n;
                }

                threads[t] = CreateThread(NULL, 0, markMultiples, &threadData[t], 0, NULL);
            }

            for (int t = 0; t < NUM_THREADS; t++) {
                WaitForSingleObject(threads[t], INFINITE);
                CloseHandle(threads[t]);
            }
        }
    }

    int count = 0;
    for (int p = 2; p <= n; p++)
        if (prime[p]) count++;

    printf("count %d\n ", count);

    free(prime);
}


int main() {
    int n = 1000000000; // Large number to test the CPU
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
