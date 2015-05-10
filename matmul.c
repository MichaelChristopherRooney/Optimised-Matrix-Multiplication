#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

/* create new empty matrix */
double ** createEmptyMatrix(int dimOne, int dimTwo){

	double ** result = malloc(sizeof(double*) * dimOne);
	double * new_matrix = malloc(sizeof(double) * dimOne * dimTwo);
	
	int i;
	for ( i = 0; i < dimOne; i++ ) {
		result[i] = &(new_matrix[i*dimTwo]);
	}

	return result;

}

/* create a new matrix and fill it with random values */
double ** generateRandomMatrix(int dimOne, int dimTwo){

	double ** temp = createEmptyMatrix(dimOne, dimTwo);

	srandom(time(NULL));

	int i, n;
	for(i = 0; i < dimOne; i++){
		for(n = 0; n < dimTwo; n++){
			temp[i][n] = random();
		}
	}

	return temp;

}

/* multiply matrices with no optimisations */
void multiplyMatices(double ** A, double ** B, double ** result, int aDim, int sharedDim, int bDim){

	int i, j, k;
	for(i = 0; i < aDim; i++){
		for(j = 0; j < bDim; j++){

			double sum = 0.0;
			for(k = 0; k < sharedDim; k++){
				sum += A[i][k] * B[k][j];
			}

			result[i][j] = sum;

		}
	}

}

/* transpose a matrix to optimise cache usage - see README for details */
double ** transpose(double ** p, int dimOne, int dimTwo){

	double ** t = createEmptyMatrix(dimTwo, dimOne);

	int i, j;
	for(i = 0; i < dimOne; i++){
		for(j = 0; j < dimTwo; j++){
			t[j][i] = p[i][j];
		}
	}

	return t;

}

/* multiply matrices using OpenMP and transposing */
void multiplyMatricesOptimised(double ** A, double ** B, double ** result, int aDim, int sharedDim, int bDim){

	B = transpose(B, sharedDim, bDim);

	int i;
	#pragma omp parallel for
	for(i = 0; i < aDim; i++){

		int j, k;
		for(j = 0; j < bDim; j++){

			double sum = 0.0;
			for(k = 0; k < sharedDim; k++){
				// B[j][k] instead of B[k][j] as B has been transposed
				sum += A[i][k] * B[j][k]; 
			}

			result[i][j] = sum;

		}

	}

}	

/* make the sure the result from the optimised multiplication matches the control */
void checkResults(double ** rOne, double ** rTwo, int dimOne, int dimTwo){

	int i, j;
	int error = 0;

	for(i = 0; i < dimOne; i++){
		for(j = 0; j < dimTwo; j++){

			if(rOne[i][j] != rTwo[i][j]){
				error = 1;
			}

		}
	}

	if(error){
		printf("ERROR: Matrices do not match\n");
	}else{
		printf("OK: Matrices match\n");
	}

}

int main(int argc, char** argv){

	int aDimOne, aDimTwo, bDimOne, bDimTwo;
	struct timeval startTime; struct timeval stopTime;
	long long originalTime = 0L; long long optimisedTime = 0L;

	if(argc != 5){
		printf("Error: parameters not provided.\n");
		return -1;
	}

	aDimOne = atoi(argv[1]);
	aDimTwo = atoi(argv[2]);
	bDimOne = atoi(argv[3]);
	bDimTwo = atoi(argv[4]);

	if(aDimTwo != bDimOne){
		printf("Number of columns in A does not match number of rows in B\n");
		return -1;
	}

	double ** A = generateRandomMatrix(aDimOne, aDimTwo);
	double ** B = generateRandomMatrix(bDimOne, bDimTwo);
	double ** originalResult = createEmptyMatrix(aDimOne, bDimTwo);
	double ** optimisedResult = createEmptyMatrix(aDimOne, bDimTwo);

	// time unoptimised multiplication
	gettimeofday(&startTime, NULL);
	multiplyMatices(A, B, originalResult, aDimOne, aDimTwo, bDimTwo);
	gettimeofday(&stopTime, NULL);
	originalTime += (stopTime.tv_sec - startTime.tv_sec) * 1000000L + (stopTime.tv_usec - startTime.tv_usec);


	// time optimised multiplication
	gettimeofday(&startTime, NULL);
	multiplyMatricesOptimised(A, B, optimisedResult, aDimOne, aDimTwo, bDimTwo);
	gettimeofday(&stopTime, NULL);
	optimisedTime = (stopTime.tv_sec - startTime.tv_sec) * 1000000L + (stopTime.tv_usec - startTime.tv_usec);

	checkResults(originalResult, optimisedResult, aDimOne, bDimTwo);

	printf("Unoptimised multiplication took: %lld microseconds\n", originalTime / 10);
	printf("Optimised multiplication took: %lld microseconds\n", optimisedTime / 10);

}
