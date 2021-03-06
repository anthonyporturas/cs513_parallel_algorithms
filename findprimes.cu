#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "cuda_runtime.h"

// Interleaved Parallel Sieve of Eratosthenes
// O(sqrt(n)loglog(n))???
// Skips several sqrt(n)
// Way to improve to allow for more processors to fill in more gaps?

__global__ void parInterweavedSOE(int * a, int n, int sqrt_N) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		a[0] = 1;
		a[1] = 1;
	}
	if (index * index < n + 1) {
		for (int i = 2; i * i <= n; i++) {
			if (a[i] != 1) {
				for (int j = 2; (index + j)*i < n + 1; j += sqrt_N) {
					if (index + j >= i) {
						a[(index + j) * i] = 1;
					}
				}
			}
		}
	}
}

// Parallel version of the Sieve of Eratosthenes
// Naive Implementation

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


// Sequential version of the Sieve of Eratosthenes
// O(nloglog(n))
void seqSOE(int * a, int n) {
	a[0] = 1;
	for (int i = 2; i * i <= n; i++) {
		if (a[i] != 1) {
			for (int j = i * 2; j <= n; j += i) {
				a[j] = 1;
			}
		}
	}
}

// Compares the parallel implementation to the sequential implementation
// to check for correctness
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

// Creates the prime testing array from 0 to N
int * createArray(int n) {
	int * a = (int*)malloc((n + 1) * sizeof(int));
	for (int i = 0; i <= n; i++) {
		a[i] = i;
	}
	return a;
}

// Prints the prime numbers from the input array
// Checks to see if value is 1, which indicates not prime
void printPrimes(int * a, int n) {

	for (int i = 0; i <= n; i++) {
		if (a[i] != 1) {
			std::cout << a[i] << " ";
		}
	}

	std::cout << std::endl;
}

int main(int argc, const char * argv[]) {

	if (argc != 2) {
		printf("The argument is wrong! Execute your program with a size n\n");
		return 1;
	}

	int n = atoi(argv[1]);
	int sqrt_N = int(ceil(sqrt(n)));
	int threads = 1024;
	int blocks = 32;
	clock_t start1, start2, end1, end2;
	int diff, diff2;

	int * a = createArray(n);
	int * b = createArray(n);
	int * d_a;
	cudaMalloc(&d_a, (n + 1) * sizeof(int));
	cudaMemcpy(d_a, a, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);


	//printPrimes(a, n);

	start1 = clock();

	seqSOE(b, n);

	end1 = clock();
	diff = (end1 - start1) * 1000 / CLOCKS_PER_SEC;
	printf("-----------------------------------\n");
	printf("Time taken to run the sequential algorithm: %d ms\n", diff);

	//printPrimes(b, n);

	//validate(a, b, n);

	start1 = clock();

	// parSOE << <blocks, threads >> > (d_a, n);
	parInterweavedSOE << <blocks, threads >> > (d_a, n, sqrt_N);
	cudaDeviceSynchronize();

	end1 = clock();
	diff = (end1 - start1) * 1000 / CLOCKS_PER_SEC;
	printf("-----------------------------------\n");
	printf("Time taken to run the parallel algorithm: %d ms\n", diff);

	//printPrimes(a, n);
	start2 = clock();
	cudaMemcpy(a, d_a, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	end2 = clock();
	diff2 = (end2 - start2) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken to copy memory from GPU onto CPU: %d ms\n", diff2);
	printf("Total: %d ms\n", diff + diff2);
	printf("-----------------------------------\n");
	//printPrimes(a, n);

	validate(a, b, n);

	cudaFree(d_a);

	free(a);
	free(b);


	return 0;
}
