#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "cuda_runtime.h"


__global__ void parInterweavedSOE(int * a, int n, int sqrt_N) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		a[0] = 1;
		a[1] = 1;
	}
	if (index * index <= n + 1) {
		for (int i = 2; i * i <= n + 1; i++) {
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


// Create random matrices given row and column sizes
// Right now for testing, only fills in incremented values
// Needs to be float values
int ** randMat(int row, int col) {
	int i, j, count;
	int **arr = (int **)malloc(row * sizeof(int *));
	for (i = 0; i<row; i++)
		arr[i] = (int *)malloc(col * sizeof(int));

	// Note that arr[i][j] is same as *(*(arr+i)+j)
	count = 0;
	for (i = 0; i < row; i++)
		for (j = 0; j < col; j++)
			arr[i][j] = ++count;  // OR *(*(arr+i)+j) = ++count

	return arr;
}

// The sequential matrix multiplication version
// Runs in O(lmnd)
// Where l is the number of rows in the first matrix
// m is the number of columns in the first matrix
// n is the number of columns in the second matrix
// d is the number of matrices to multiply
int ** seqMult(int * dimVect, int dimNum, int ***b) {
	int i, j, k;
	int **output = b[0];
	int **mat1;
	int **mat2;
	for (k = 0; k < dimNum - 3; k++) {
		int tempRow;
		if (k == 0) {
			mat1 = b[k];
			tempRow = dimVect[k];
		}
		else {
			mat1 = output;
		}

		mat2 = b[k + 1];

		int tempCol = dimVect[k + 2];
		int commonDim = dimVect[k + 1];
		output = (int **)malloc(tempRow * sizeof(int *));
		for (i = 0; i < dimVect[k]; i++) {
			output[i] = (int *)malloc(tempCol * sizeof(int));
		}

		for (i = 0; i < tempRow; i++) {
			for (j = 0; j < tempCol; j++) {
				int total = 0;
				for (int multNum = 0; multNum < commonDim; multNum++) {
					total += mat1[i][multNum] * mat2[multNum][j];
				}
				output[i][j] = total;
			}
		}

	}
	return output;
}

int main(int argc, const char * argv[]) {
	
	// Check to see if there is at least one matrix
	if (argc < 3) {
		printf("The argument is wrong! Execute your program with a vector with at least two integers.\n");
		return 1;
	}
	
	int i, j, k;

	// Arary holding all the dimensions
	int * dimVect = (int *)malloc((argc - 1) * sizeof(int));
	for (i = 1; i < argc; i++) {
		dimVect[i - 1] = atoi(argv[i]);
	
	}

	int **a; // Temporary 2D array to hold current matrix
	int ***b = (int ***)malloc((argc - 1) * sizeof(int**)); // Array holding all matrices
	for (i = 0; i < argc-2; i++) {
		a = randMat(dimVect[i], dimVect[i+1]);
		b[i] = a;
	}


	// Print out all matrices
	for (k = 0; k < argc - 2; k++) {
		a = b[k];
		int tempRow = dimVect[k];
		int tempCol = dimVect[k + 1];
		for (i = 0; i < tempRow; i++)
			for (j = 0; j < tempCol; j++)
				printf("%d ", a[i][j]);
		printf("\n");
	}


	// Call Sequential Multiplication
	int **result = seqMult(dimVect, argc, b);

	
	// Check Results
	for (i = 0; i < dimVect[0]; i++)
		for (j = 0; j < dimVect[argc-2]; j++)
			printf("%d ", result[i][j]);
	printf("\n");

	free(a);
	free(dimVect);
	free(b);
	return 0;	
}
