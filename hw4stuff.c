#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "cuda_runtime.h"
#include <math.h>


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
__device__ void seqMult2(int ** output, int tempRow, int commonDim, int tempCol, int **mat1, int **mat2) {
	int i, j;

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
__global__ void parMat(int * dimVect, int numDim, int *** matList, int *** intermediate, int levels) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int ind = 2;
	int threadMax = ((numDim - 2) / 2) + ((numDim - 2) % 2);
	
	int tempRow, tempCol, commonDim;
	for (int i = 0; i < levels; i++) {
		if (index < threadMax) {

			if (i == 0) {
				tempRow = dimVect[index*2-1];
				tempCol = dimVect[index*2 + 1];
				commonDim = dimVect[index*2];
				int **mat1 = matList[index * 2 - 1];
				int **mat2 = matList[index * 2];
				int **output = (int **)malloc(tempRow * sizeof(int *));
				for (int k = 0; k < tempRow; k++) {
					output[k] = (int *)malloc(tempCol * sizeof(int));
				}
				seqMult2(output, tempRow, commonDim, tempCol, mat1, mat2);
				intermediate[index] = output;
			}
			else {
				commonDim = tempCol;
				tempCol = dimVect[index * 2 + (i * 2)];
				int **mat1 = intermediate[index];
				int **mat2 = intermediate[index + 1];
				int **output = (int **)malloc(tempRow * sizeof(int *));
				for (int k = 0; k < tempRow; k++) {
					output[k] = (int *)malloc(tempCol * sizeof(int));
				}
				__syncthreads();
				seqMult2(output, tempRow, commonDim, tempCol, mat1, mat2);
				intermediate[index] = output;
			}
			__syncthreads();
		}
		ind = ind * 2;
		threadMax = ((threadMax - 2) / 2) + ((threadMax - 2) % 2);
	}
	

}


__global__ void parMult(int** output, int **mat1, int **mat2, int row, int com, int col) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int i, j;
	
		output = (int **)malloc(row * sizeof(int *));
		for (i = 0; i < row; i++) {
			output[i] = (int *)malloc(col * sizeof(int));
		}

		i = index / col;
		j = (index % col) - 1;
		int total = 0;
		for (int multNum = 0; multNum < com; multNum++) {
			total += mat1[i][multNum] * mat2[multNum][j];
		}

		output[i][j] = total;
			
		

}


// Create random matrices given row and column sizes
// Right now for testing, only fills in incremented values
// Needs to be int values
int ** randMat(int row, int col) {
	int i, j;
	int count;
	int **arr = (int **)malloc(row * sizeof(int *));
	for (i = 0; i<row; i++)
		arr[i] = (int *)malloc(col * sizeof(int));

	// Note that arr[i][j] is same as *(*(arr+i)+j)
	count = 0;
	for (i = 0; i < row; i++) {
		for (j = 0; j < col; j++) {
			//printf("%d ", count);
			//printf("\n");
			arr[i][j] = ++count;  // OR *(*(arr+i)+j) = ++count
			//printf("%d ", arr[i][j]);
		}
	}

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
	int threads = 1024; // Standard number of threads
	int blocks = 32; // Magic value that results in best running speeds and larger acceptable inputs n

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
	
	//int ***intermediate = (int ***)malloc((argc - 1) * sizeof(int**));

	// Call Sequential Multiplication
	int **result = seqMult(dimVect, argc, b);
	for (i = 0; i < dimVect[0]; i++)
		for (j = 0; j < dimVect[argc - 2]; j++)
			printf("%d ", result[i][j]);
	printf("\n");
	int levels = log(argc - 2);

	int * d_dimVect;
	cudaMalloc(&d_dimVect, (argc-1) * sizeof(int));
	cudaMemcpy(d_dimVect, dimVect, (argc-1) * sizeof(int), cudaMemcpyHostToDevice);

	int *** matList;
	cudaMalloc(&matList, (argc - 2) * sizeof(int**));
	cudaMemcpy(matList, b, (argc - 2) * sizeof(int**), cudaMemcpyHostToDevice);

	int ***out = (int ***)malloc((((argc - 2) / 2) + 1) * sizeof(int**));
	

	int *** intermediate;
	cudaMalloc(&intermediate, (((argc - 2)/2)+1) * sizeof(int**));
	cudaMemcpy(intermediate, out, (((argc - 2) / 2) + 1) * sizeof(int**), cudaMemcpyHostToDevice);
	/*
	for (int z = 0; z < (((argc - 2) / 2) + 1); z++) {
		cudaMalloc(&intermediate[z], (((argc - 2) / 2) + 1) * sizeof(int*));
	}
	*/
	parMat << <blocks, threads >> > (d_dimVect, argc, matList, intermediate, levels);
	
	cudaMemcpy(out, intermediate, (argc-2) * sizeof(int**), cudaMemcpyDeviceToHost);

	cudaFree(d_dimVect);
	cudaFree(matList);
	cudaFree(intermediate);
	
	int **result2 = out[0];
	//printf("%d \n", result[0][0]);
	/**************************************
	
	int **output = b[0];
	int **mat1;
	int **mat2;
	int **d_mat1, **d_mat2;
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

		cudaMalloc(&d_output, tempRow * sizeof(int*));
		cudaMalloc(&d_mat1, tempRow * sizeof(int*));
		cudaMalloc(&d_mat2, commonDim * sizeof(int*));
		cudaMemcpy(d_mat1, mat1, tempRow * sizeof(int*), cudaMemcpyHostToDevice);
		cudaMemcpy(d_mat2, mat2, commonDim * sizeof(int*), cudaMemcpyHostToDevice);
		parMult << <blocks, threads >> > (d_a, n, sqrt_N);

	}
	
	
	cudaMemcpy(b, d_b, (argc-1) * sizeof(int**), cudaMemcpyDeviceToHost);
	***********************************************/

	// Check Results
	
	for (i = 0; i < dimVect[0]; i++)
		for (j = 0; j < dimVect[argc-2]; j++)
			printf("%d ", result2[i][j]);
	printf("\n");
	
	//cudaFree(d_b);
	
	return 0;	
}
