/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)


/* write matrix to stdout */
void write_out(double ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%f, ", a[i][j]);
    }
    printf("%f\n", a[i][dim2-1]);
  }
}


/* create new empty matrix */
double ** new_empty_matrix(int dim1, int dim2)
{
  double ** result = malloc(sizeof(double*) * dim1);
  double * new_matrix = malloc(sizeof(double) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}

/* take a copy of the matrix and return in a newly allocated matrix */
double ** copy_matrix(double ** source_matrix, int dim1, int dim2)
{
  int i, j;
  double ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
double ** gen_random_matrix(int dim1, int dim2)
{
  double ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      long long upper = random();
      long long lower = random();
      result[i][j] = (double)((upper << 32) | lower);
    }
  }

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(double ** result, double ** control, int dim1, int dim2)
{
  int i, j;
  int error = 0;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
	if(control[i][j] != result[i][j]){
		error = 1;
	}
    }
  }

	if(error){
		printf("Matrices do not match\n");
	}

}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(double ** A, double ** B, double ** C, int a_dim1, int a_dim2, int b_dim2)
{
	int i, j, k;

	for(i = 0; i < a_dim1; i++) {
		for( j = 0; j < b_dim2; j++ ) {
			double sum = 0.0;
				for(k = 0; k < a_dim2; k++) {
					sum += A[i][k] * B[k][j];
				}
			C[i][j] = sum;
		}
	}

}

/* transposing the second matrix can speed up multiplication - see report */
double** transpose(double ** B, int a_dim2, int b_dim2){

        double ** trB = new_empty_matrix(b_dim2, a_dim2);
        int i, j;

        for(i = 0; i < a_dim2; i++){
                for(j = 0; j < b_dim2; j++){
                        trB[j][i] = B[i][j];
                }
        }

        return trB;

}

/* the fast version of matmul written by the team */
void team_matmul(double ** A, double ** B, double ** C, int a_dim_1, int a_dim_2, int b_dim_2){

	/* create local copies to allow them to be stored in registers */
	int a_dim1 = a_dim_1;
	int a_dim2 = a_dim_2;
	int b_dim2 = b_dim_2;

	if((a_dim1 >= 159 || a_dim2 >= 159 || b_dim2 >= 159) && (a_dim2 / a_dim1 < 22 || a_dim2 / b_dim2 < 22)){

		/* transposing the second matrix allows more of it to be in the cache - see report for details */
		B = transpose(B, a_dim2, b_dim2);

		int i;
		#pragma omp parallel for
		for(i = 0; i < a_dim1; i++){

			int j, k;
			double sum0;

			for(j = 0; j < b_dim2; j++){

				sum0 = 0;

				for(k = 0; k < a_dim2; k++) {
					/* need to use B[j][k] instead of B[k][j] to account for the fact that the matrix is transposed */
					sum0 += A[i][k] * B[j][k];
				}

				C[i][j] = sum0;

			}

		}

	}else{

		int i, j, k;
		double sum0;

		for(i = 0; i < a_dim1; i++){

			for(j = 0; j < b_dim2; j++){

				sum0 = 0;

				for(k = 0; k < a_dim2; k++) {
					sum0 += A[i][k] * B[k][j];
				}

				C[i][j] = sum0;

			}

		}

	}

}

int main(int argc, char ** argv)
{
  double ** A, ** B, ** C;
  double ** control_matrix;
  long long mul_time;
  int a_dim1, a_dim2, b_dim1, b_dim2;
  struct timeval start_time;
  struct timeval stop_time;

  if ( argc != 5 ) {
    fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /* check the matrix sizes are compatible */
  if ( a_dim2 != b_dim1 ) {
    fprintf(stderr,
	    "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
	    a_dim2, b_dim1);
    exit(1);
  }

  /* allocate the matrices */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* use a simple matmul routine to produce control result */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Matmul time (original): %lld microseconds\n", mul_time);

  //write_out(control_matrix, a_dim1, b_dim2);
  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* perform matrix multiplication */
  team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Matmul time (optimised): %lld microseconds\n", mul_time);

  //write_out(C, a_dim1, b_dim2);

  /* now check that the team's matmul routine gives the same answer
     as the known working version */
  check_result(C, control_matrix, a_dim1, b_dim2);

  return 0;
}
