#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "cuda_runtime.h"
#include <math.h>


__device__ void seqMult2(int * output, int tempRow, int commonDim, int tempCol, int *mat1, int *mat2) {

	for (int i = 0; i < tempRow * tempCol; i++) {
		int total = 0;
		int j = i % tempCol;

		for (int k = 0; k < commonDim; k++) {
			total += mat1[k + (i / tempCol)*commonDim] * mat2[(k*tempCol + j)];
		}

		output[i] = total;

	}


}
__global__ void parMat2(int * dimVect, int numDim, int ** matList, int ** intermediate, int levels) {
	/*
	for (int i = 0; i < numDim - 2; i++) {
		intermediate[i][0] = 1;
		intermediate[i][1] = 2;
	}
	*/
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	
		int commonDim = dimVect[1];
		int tempRow = dimVect[0];
		int tempCol = dimVect[2];
		int *mat1 = matList[0];
		int *mat2 = matList[1];
		int *output = new int[tempRow*tempCol];
		seqMult2(output, tempRow, commonDim, tempCol, mat1, mat2);
		int threadMax = ((numDim - 2) / 2) + ((numDim - 2) % 2);
		threadMax = 100;
		
		if (index > threadMax) {
			intermediate[0][index % (tempRow*tempCol)] = output[index % (tempRow*tempCol)];
		}
		
		
	

}
__global__ void parMat(int * dimVect, int numDim, int ** matList, int ** intermediate, int levels) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int threadMax = ((numDim - 2) / 2) + ((numDim - 2) % 2);
	if (index == 0) {
		return;
	}
	int tempRow, tempCol, commonDim;
	for (int i = 0; i < levels; i++) {
		if (index <= threadMax) {

			if (i == 1) {
				tempRow = dimVect[index * 2 - 2];
				tempCol = dimVect[index * 2];
				commonDim = dimVect[index * 2 - 1];
				int *mat1 = matList[index * 2 - 2];
				int *mat2 = matList[index * 2 - 1];
				//int *output = (int *)malloc(tempRow * tempCol* sizeof(int ));
				int *output = new int[tempRow*tempCol];
				
				seqMult2(output, tempRow, commonDim, tempCol, mat1, mat2);
				intermediate[index] = output;
			}
			else {
				commonDim = tempCol;
				tempCol = dimVect[index * 2 + (i * 2) - 1];
				int *mat1 = intermediate[index];
				int *mat2 = intermediate[index + 1];
				//int *output = (int *)malloc(tempRow * tempCol* sizeof(int ));
				int *output = new int[tempRow*tempCol];
				__syncthreads();
				seqMult2(output, tempRow, commonDim, tempCol, mat1, mat2);
				intermediate[index] = output;
			}
			__syncthreads();
		}

		threadMax = ((threadMax - 2) / 2) + ((threadMax - 2) % 2);
	}
	__syncthreads();
	if (index == 1) {
		intermediate[0] = intermediate[1];
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
int * randMat(int row, int col) {
	int i, j;
	int count;
	int *arr = (int *)malloc(row * col* sizeof(int ));

	count = 0;
	for (i = 0; i < row*col; i++) {
			arr[i] = ++count;  
		
	}

	return arr;
}

// The sequential matrix multiplication version
// Runs in O(lmnd)
// Where l is the number of rows in the first matrix
// m is the number of columns in the first matrix
// n is the number of columns in the second matrix
// d is the number of matrices to multiply
int * seqMult(int * dimVect, int dimNum, int **b) {
	int i, j, k;
	int *output = b[0];
	int *mat1;
	int *mat2;
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
		output = (int *)malloc(tempRow * tempCol*sizeof(int ));

		for (i = 0; i < tempRow * tempCol; i++) {
				int total = 0;
				int j = i % tempCol;
				
				for (int k = 0; k < commonDim; k++) {
					total += mat1[k+(i/ tempCol)*commonDim] * mat2[(k*tempCol + j)];
				}

				output[i] = total;
			
		}

	}
	return output;
}

#include <cstdio>
inline void GPUassert(cudaError_t code, char * file, int line, bool Abort = true)
{
	if (code != 0) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (Abort) exit(code);
	}
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }



__global__ void doSmth(int*** a) {
	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++)
			for (int k = 0; k<2; k++)
				a[i][j][k] = i + j + k;
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

	int *a; // Temporary 2D array to hold current matrix
	int **b = (int **)malloc((argc - 1) * sizeof(int*)); // Array holding all matrices
	for (i = 0; i < argc - 2; i++) {
		a = randMat(dimVect[i], dimVect[i + 1]);
		b[i] = a;
	}


	// Print out all matrices
	for (k = 0; k < argc - 2; k++) {
		printf("Mat %d: ", k + 1);
		a = b[k];
		int tempRow = dimVect[k];
		int tempCol = dimVect[k + 1];
		for (i = 0; i < tempRow*tempCol; i++)
				printf("%d ", a[i]);
		printf("\n");
	}

	int *result = seqMult(dimVect, argc, b);
	printf("Sequential Result:\n");
	for (int i = 0; i < dimVect[0] * dimVect[argc - 2]; i++) {
		printf("%d ", result[i]);
	}
	printf("\n");
	printf("\n");

	int * d_dimVect;
	cudaMalloc((void **)&d_dimVect, (argc-1) * sizeof(int));
	cudaMemcpy(d_dimVect, dimVect, (argc - 1) * sizeof(int), cudaMemcpyHostToDevice);

	int ** matList = (int **)malloc((argc-2) * sizeof(int *));
	for (i = 0; i < (argc-2); i++) {
		cudaMalloc((void **)&matList[i], dimVect[i] * dimVect[i + 1] * sizeof(int));
		cudaMemcpy(matList[i], b[i], dimVect[i] * dimVect[i + 1] * sizeof(int), cudaMemcpyHostToDevice);
	}

	int ** d_matList;
	cudaMalloc(&d_matList, (argc - 2) * sizeof(int *));
	cudaMemcpy(d_matList, matList, (argc - 2) * sizeof(int *), cudaMemcpyHostToDevice);


	int ** interMat = (int **)malloc(ceil((argc - 2) / 2) + 1 * sizeof(int *));
	int * currentMat = (int *)calloc(100, sizeof(int));
	for (i = 0; i < ceil((argc-2)/2)+1; i++) {
		cudaMalloc((void **)&interMat[i], 100 * sizeof(int));
		cudaMemcpy(interMat[i], currentMat, 100 * sizeof(int), cudaMemcpyHostToDevice);
	}

	int ** d_interMat;
	cudaMalloc(&d_interMat, (argc - 2) * sizeof(int *));
	cudaMemcpy(d_interMat, interMat, (argc - 2) * sizeof(int *), cudaMemcpyHostToDevice);

	int levels = log(argc - 2) / log(2);
	parMat2 << <blocks, threads >> > (d_dimVect, argc, d_matList, d_interMat, levels);

	/*
	int *res = (int *)malloc(100 * 100 * sizeof(int));
	for (i = 0; i < ceil((argc - 2) / 2) + 1; i++) {
		printf("1:\n");
		cudaMemcpy(res, d_interMat[i], 100 * 100 * sizeof(int), cudaMemcpyDeviceToHost);
		printf("2:\n");
		b[i] = res;
	}
	*/

	int res[2][4];
	for (int i = 0; i<4; i++)
			cudaMemcpy(&res[i][0], interMat[i], 16*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i<2; i++)
		for (int j = 0; j<4; j++)
				printf("[%d][%d]=%d\n", i, j, res[i][j]);
	return 0;
}
