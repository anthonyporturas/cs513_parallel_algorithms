#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "cuda_runtime.h"

//
// Interleaved Parallel Sieve of Eratosthenes
// O(sqrt(n)loglog(n))
// Each skip is multiplied by a factor of sqrt(n). This reduces the running time of each processor
// by a factor of sqrt(n). The rest of the of the sqrt(n) processors fill in the gaps in parallel.
// This effectively reduces the total running time by a factor of sqrt(n).
// 
__global__ void parInterweavedSOE(int * a, int n, int sqrt_N) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        a[0] = 1;
        a[1] = 1;
    }
    if (index * index <= n + 1) {
        for (int i = 2; i * i <= n+1; i++) {
            if (a[i] != 1) {
                for (int j = 2; (index + j)*i < n + 1; j += sqrt_N) {
                    if (index + j >= i) {
                        a[(index + j) * i] = 1;
                    }
                }
                __syncthreads();
            }
        }
    }
}

//
// Naive Implementation of Parallel version of the Sieve of Eratosthenes
// Not interleaved, lots of overlapping work done. Inefficient
//

__global__ void parSOE(int * a, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        a[0] = 1;
        a[1] = 1;
    }

    if (index * index <= n + 1 && a[index] != 1 && index > 1) {
        for (int i = index * index; i <= n + 1; i += index) {
            a[i] = 1;
        }
    }
}

//
// Sequential version of the Sieve of Eratosthenes
// O(nloglog(n))
// This was used as the base line for improvement and is used as creating the a workable
// and correct array for comparison.
//
void seqSOE(int * a, int n) {
    a[0] = 1;
    for (int i = 2; i * i <= n+1; i++) {
        if (a[i] != 1) {
            for (int j = i * i; j <= n+1; j += i) {
                a[j] = 1;
            }
        }
    }
}

//
// Used to compare the parallel implementation to the sequential implementation
// to check for correctness
//
void validate(int * a, int * b, int n) {
    int count = 0;
    for (int i = 0; i <= n; i++) {
        if (a[i] != b[i]) {
            count++;
        }
    }

    std::cout << "Correct: " << (n + 1) - count << std::endl;
    std::cout << "Total: " << n + 1 << std::endl;
}

//
// Creates the prime array from 0 to N
// We decided to use the indices to indicate the corresponding number
// E.g. a[0] = 0, a[1] = 1, a[2] = 2, ..., a[n] = n
//
int * createArray(int n) {
    int * a = (int*)malloc((n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) {
        a[i] = i;
    }
    return a;
}

//
// Prints the prime numbers in the given array
// Checks to see if the value is 1, which indicates it is not a prime number
//
void printPrimes(int * a, int n) {

    for (int i = 0; i <= n; i++) {
        if (a[i] != 1) {
            std::cout << a[i] << " ";
        }
    }

    std::cout << std::endl;
}

//
// Run with a valid integer between 0 and 500 million
//
int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("The argument is wrong! Execute your program with a size n\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int sqrt_N = int(ceil(sqrt(n)));
    int threads = 1024; // Standard number of threads
    int blocks = 32; // Magic value that results in best running speeds and larger acceptable inputs n
    clock_t start1, start2, start3, end1, end2, end3;
    int diff1, diff2, diff3;

    int * a = createArray(n);
    int * b = createArray(n);
    int * d_a;
    cudaMalloc(&d_a, (n + 1) * sizeof(int));
    


    //printPrimes(a, n);

    start1 = clock();

    seqSOE(b, n);

    end1 = clock();
    diff1 = (end1 - start1) * 1000 / CLOCKS_PER_SEC;
    printf("-----------------------------------\n");
    printf("Time taken to run the sequential algorithm: %d ms\n", diff1);

    //printPrimes(b, n);

    //validate(a, b, n);

    start1 = clock();
    cudaMemcpy(d_a, a, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    end1 = clock();
    diff1 = (end1 - start1) * 1000 / CLOCKS_PER_SEC;

    //parSOE << <blocks, threads >> > (d_a, n);
    start2 = clock();
    parInterweavedSOE << <blocks, threads >> > (d_a, n, sqrt_N);
    cudaDeviceSynchronize();
    end2 = clock();
    diff2 = (end2 - start2) * 1000 / CLOCKS_PER_SEC;
    printf("-----------------------------------\n");
    printf("Time taken to run the parallel algorithm: %d ms\n", diff2);

    //printPrimes(a, n);
    start3 = clock();
    cudaMemcpy(a, d_a, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    end3 = clock();
    diff3 = (end3 - start3) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken to copy memory from CPU onto GPU: %d ms\n", diff1);
    printf("Time taken to copy memory from GPU onto CPU: %d ms\n", diff3);
    printf("Total: %d ms\n", diff1 + diff2 + diff3);
    printf("-----------------------------------\n");
    //printPrimes(a, n);

    validate(a, b, n);

    cudaFree(d_a);

    free(a);
    free(b);


    return 0;
}

