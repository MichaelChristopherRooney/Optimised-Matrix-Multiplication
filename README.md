#About


A program to speed up matrix multiplication.


Uses OpenMP to compute results in parallel, loop unrolling to reduce overhead and transposing to optimise cache usage (see http://www.akkadia.org/drepper/cpumemory.pdf Section 6.2.1)


The dimensions of the matrices are passed as arguments to the program, in the form: "./matmul -A rows -A cols -B rows -B cols"


#Timings

Timed using an Intel i5 2500k CPU (http://ark.intel.com/products/52210/Intel-Core-i5-2500K-Processor-6M-Cache-up-to-3_70-GHz) with turbo frequency disabled to ensure fair timing.


Each category was run 10 times and the given result is an average of the timings.

In all categories two 1000x1000 matrices were multiplied.

Compiled with "gcc -fopenmp matmul.c"

	Normal: 10.31 seconds
	Using loop unrolling: 9.6 seconds
	Using a transposed matrix: 3.53 seconds
	Using OpenMP: 2.91 seconds
	Using OpenMP and a transposed matrix: 1.13 seconds
	Using OpenMP, a transposed matrix and loop unrolling: 0.93 seconds

These numbers show that the optimised version on average runs 11x faster.

#Issues

The optimised function will be slower than the unoptimised function when dealing with very small matrices.

This is due to the overhead of creating threads and transposing the matrix being larger than the time saved.

As a protection against this, if both matrices are smaller than 50x50 they will be multiplied by the unoptimised function.
