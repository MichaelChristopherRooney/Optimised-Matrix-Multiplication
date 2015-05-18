#About


A program to speed up matrix multiplication.


Uses OpenMP to compute results in parallel, loop unrolling to reduce overhead and pipeline flushes and transposing to optimise cache usage (see http://www.akkadia.org/drepper/cpumemory.pdf Section 6.2.1)


The dimensions of the matrices are passed as arguments to the program, in the form: "./matmul -A rows -A cols -B rows -B cols"


#Timings

This was program was timed on two machines. Several categories were timed to measure how much each optimisation decreased the time taken. Each category was run 10 times and the given result is an average of the timings. In all categories two 1000x1000 matrices were multiplied. 

The program was compiled with 
"gcc -fopenmp matmul.c"

Using an Intel i5 2500k CPU (http://ark.intel.com/products/52210/Intel-Core-i5-2500K-Processor-6M-Cache-up-to-3_70-GHz) with turbo frequency disabled to ensure fair timing.

	Normal: 10.31 seconds
	Using loop unrolling: 9.6 seconds
	Using a transposed matrix: 3.53 seconds
	Using OpenMP: 2.91 seconds
	Using OpenMP and a transposed matrix: 1.13 seconds
	Using OpenMP, a transposed matrix and loop unrolling: 0.93 seconds

Using an AMD 64 core CPU

	Normal: 11.938 seconds
	Using OpenMP: 0.461 seconds
	Using OpenMP and a transposed matrix: 0.087 seconds
	
On the AMD 64 core CPU, OpenMP and transposing result in a 137x speedup. On the Intel 4 core CPU, OpenMP, transposing and loop unrolling results in a 11x speedup. Clearly this type of program benefits massively from having work done concurrently.

Note that I no longer had access to the 64 core machine when I added in loop unrolling, so I was unable to get timings for it.

#Issues

The optimised function will be slower than the unoptimised function when dealing with very small matrices.

This is due to the overhead of creating threads and transposing the matrix being larger than the time saved.

As a protection against this, if both matrices are smaller than 50x50 they will be multiplied by the unoptimised function.
