#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

void sieveOfEratosthenes(int n) {
    // Create a boolean array "prime[0..n]" and initialize
    // all entries it as true. A value in prime[i] will
    // finally be false if i is Not a prime, else true.
    bool* prime = (bool *)malloc((n + 1) * sizeof(bool));
    for (int i = 0; i <= n; i++)
        prime[i] = true;

    int sq = sqrt(n);

    for (int p = 2; p <= sq; p++) {
        // If prime[p] is not changed, then it is a prime
        if (prime[p] == true) {
            // Update all multiples of p to false
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }

    // Print prime numbers up to 100
    //for (int p = 2; p <= 100; p++)
    //    if (prime[p])
     //       printf("%d ", p);

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
