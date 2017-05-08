#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MOD 65536
#define NELE(x) (sizeof(x) / sizeof(x[0]))

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
	/*
	if (index > threadMax) {
		intermediate[0][index % (tempRow*tempCol)] = output[index % (tempRow*tempCol)];
	}
	*/
	intermediate[0][0] = output[0];
	intermediate[0][1] = output[1];
	intermediate[0][2] = output[2];
	intermediate[0][3] = output[3];

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
                total += (mat1[k + (i / tempCol)*commonDim] * mat2[(k*tempCol + j)]) % MOD;
            }

            output[i] = total % MOD;

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

	int res[2][2];
	for (int i = 0; i<2; i++)
			cudaMemcpy(&res[i][0], interMat[i], 4*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i<2; i++)
		for (int j = 0; j<4; j++)
				printf("[%d][%d]=%d\n", i, j, res[i][j]);
	
}
