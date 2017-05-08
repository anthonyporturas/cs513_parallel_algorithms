#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MOD 65536
#define NELE(x) (sizeof(x) / sizeof(x[0]))


// double * createMatrix(int row, int col){
//     double * a = (double *)malloc(row*col * sizeof(double));
//     for (int i = 0; i < row * col; i++) {
//         a[i] = 2.0;
//     }
//     return a;
// }

// void printMatrix(double * matrix, int r, int c) {
//     for(int i = 0; i < r; i++){
//         for (int j = 0; j < c; j++) {
//             printf("%f", matrix[i*r + j]);
//         }
//         printf("\n");
//     }
// }


int * createMatrix(FILE* file, int row, int col) {

	int i, j;
	int count;
	int *arr = (int *)malloc(row * col * sizeof(int));

	count = 0;
	for (i = 0; i < row*col; i++) {
		fscanf(file, "%d", &arr[i]);

	}

	return arr;
}

// The sequential matrix multiplication version
// Runs in O(lmnd)
// Where l is the number of rows in the first matrix
// m is the number of columns in the first matrix
// n is the number of columns in the second matrix
// d is the number of matrices to multiply
int ** seqMult(int * dimVect, int dimVectLen, int ***b) {
	int i, j, k;
	int **output = b[0];
	int **mat1;
	int **mat2;
	printf("AAA\n");
	for (k = 0; k < dimVectLen - 2; k++) {
		printf("k: %d\n", k);

		// printf("AAA\n");
		int tempRow;
		if (k == 0) {
			mat1 = b[k];
			tempRow = dimVect[k];
			// printf("BBB\n");
		}
		else {
			mat1 = output;
		}
		// printf("CCC\n");
		mat2 = b[k + 1];

		int tempCol = dimVect[k + 2];
		int commonDim = dimVect[k + 1];
		output = (int **)malloc(tempRow * sizeof(int *));
		// printf("DDD\n");
		for (i = 0; i < dimVect[k]; i++) {
			output[i] = (int *)malloc(tempCol * sizeof(int));
		}
		// printf("EEE\n");
		for (i = 0; i < tempRow; i++) {
			for (j = 0; j < tempCol; j++) {
				int total = 0;
				// printf("FFF\n");
				for (int multNum = 0; multNum < commonDim; multNum++) {
					// printf("GGGG\n");
					total += (mat1[i][multNum] * mat2[multNum][j]) % MOD;
				}
				// printf("HHHH\n");
				// printf("%d\n", total);
				// printf("i: %d\n", i);
				// printf("j: %d\n", j);
				output[i][j] = total % MOD;
				// output[i][j] = 0;
				// printf("IIIII\n");
			}
			printf("i: %d\n", i);
			printf("row: %d\n", tempRow);
			printf("col: %d\n", tempCol);
		}

	}
	return output;
}

int * seqMult2(int * dimVect, int dimNum, int **b) {
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
		output = (int *)malloc(tempRow * tempCol * sizeof(int));

		for (i = 0; i < tempRow * tempCol; i++) {
			int total = 0;
			int j = i % tempCol;

			for (int k = 0; k < commonDim; k++) {
				total += mat1[k + (i / tempCol)*commonDim] * mat2[(k*tempCol + j)];
			}

			output[i] = total;

		}

	}
	return output;
}

int main(int argc, const char * argv[]) {

	// Check to see if there is at least one matrix
	int i, j, k;
	int threads = 1024; // Standard number of threads
	int blocks = 32; // Magic value that results in best running speeds and larger acceptable inputs n
	printf("YYYYYYYYYYYYY\n");
	char const* const fileName = argv[1];
	FILE* file = fopen(fileName, "r");
	char buffer[2048];
	int temp_dim;
	printf("YYYYYYYYYYYYY\n");
	// replace argc - 1 with dimVectLen
	int dimVectLen = atoi(fgets(buffer, 2048, file));
	printf("YYYYYYYYYYYYY\n");
	// Arary holding all the dimensions
	int * dimVect = (int *)malloc(dimVectLen * sizeof(int));
	for (int i = 0; i < dimVectLen; i++) {
		fscanf(file, "%d", &dimVect[i]);
	}
	printf("YYYYYYYYYYYYY\n");
	int *a; // Temporary 2D array to hold current matrix
	int **b = (int **)malloc((dimVectLen) * sizeof(int*)); // Array holding all matrices
	for (i = 0; i < dimVectLen - 1; i++) {
		a = createMatrix(file, dimVect[i], dimVect[i + 1]);
		b[i] = a;
	}


	// for (k = 0; k < dimVectLen - 1; k++) {
	//         a = b[k];
	//         int tempRow = dimVect[k];
	//         int tempCol = dimVect[k + 1];
	//         for (i = 0; i < tempRow; i++)
	//             for (j = 0; j < tempCol; j++)
	//                 printf("%d ", a[i][j]);
	//         printf("\n");
	// }
	printf("YYYYYYYYYYYYY\n");
	// while(fgets(line, sizeof(line), file)) {

	// }

	// Call Sequential Multiplication
	int *result = seqMult2(dimVect, dimVectLen, b);
	printf("ZZZZZZZZZZZZ\n");

	

	for (int i = 0; i < dimVect[0] * dimVect[dimVectLen - 1]; i++) {
		printf("%d ", result[i]);
	}

	printf("\n");
	printf("success!!!!\n");
	fclose(file);
	// // srand(time(0));
	// int matrixDims[] = {3,3,3};
	// int numOfMatrixes = NELE(matrixDims) - 1;

	// double ** matrixArray = (double **)malloc(numOfMatrixes * sizeof(double *));
	// int max = 3;

	// for (int i = 0; i < numOfMatrixes; i++) {
	//     matrixArray[i] = createMatrix(matrixDims[i], matrixDims[i + 1]);
	// }

	// printMatrix(matrixArray[0], 3, 3);
}
