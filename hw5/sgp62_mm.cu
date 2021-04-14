//Stefen Pegels sgp62
// /usr/local/cuda-11.2/bin/nvcc -arch=compute_35 -o sgp62_mm.out sgp62_mm.cu
/* naive matrix matrix multiplication of two N by N matrices    */
/* 2D grid of 2D blocks where block is BLOCK_SIZE by BLOCK_SIZE */
/* and grid is N/BLOCK_SIZE by BLOCK_SIZE                       */

#include <stdio.h>
#define BLOCK_SIZE 64 //Invariant: Tile size is always equal to block size
#define N 16384

__global__ void matrixMul(float *dev_A, float * dev_B, float * dev_C, int n)
{
	float partial = 0.0;
	int k;
	int i = blockIdx.y * blockDim.y + threadIdx.y; // Row i of C
	int j = blockIdx.x * blockDim.x + threadIdx.x; // Column j of C

	for (k = 0; k < n; k++)
	  partial += dev_A[i*n + k]*dev_B[k*n + j];// not coalesed for B
  	dev_C[i*n + j] = partial;
}

__global__ void TilematrixMul(float *dev_A, float * dev_B, float * dev_C, int n, int k) //k is tile size
{
	__shared__ float cacheA[BLOCK_SIZE*BLOCK_SIZE]; //Shared mem allocation for tile
	__shared__ float cacheB[BLOCK_SIZE*BLOCK_SIZE];

	float partial = 0.0; //Temp for accumulating partial sum

	int tix = threadIdx.x;
	int tiy = threadIdx.y;
	int bix = blockIdx.x;
	int biy = blockIdx.y;

	// k == blockDim.y == blockDim.x
	int i = biy * k + tiy; // Row
	int j = bix * k + tix; // Col

	for(int m = 0; m < n/k; m++){ //Loop over tiles
		cacheA[tiy*k + tix] = dev_A[i*n + (m*k + tix)]; //Loading a tile
		cacheB[tiy*k + tix] = dev_B[(m*k*n + tiy*n) + j];
		__syncthreads();
		for(int l = 0; l < k; l++){
			partial += cacheA[tiy*k + l] * cacheB[l*k + tix];//accumulate temp
		}
		__syncthreads();
	}
	dev_C[i*n + j] = partial; //Store in result C
}

int main(){
    float *A, *B, *C, *dev_A, *dev_B, *dev_C;
	int i;
	long NN = N;

	A = (float*)malloc( N*N*sizeof(float) );
	cudaMalloc( (void**)&dev_A, N*N*sizeof(float));
	for (i = 0; i < N*N; i++) A[i] = 1.0/(1.0+i);

	B = (float*)malloc( N*N*sizeof(float) );
	cudaMalloc( (void**)&dev_B, N*N*sizeof(float));
	for (i = 0; i < N*N; i++) B[i] = i+1.0;

	C = (float*)malloc( N*N*sizeof(float) );
	cudaMalloc( (void**)&dev_C, N*N*sizeof(float));

	dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 Grid(N/Block.x, N/Block.y);

// timing variables
	cudaEvent_t start, stop;
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 
	
// start timing
    cudaEventRecord( start, 0 );

	cudaMemcpy(dev_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B ,N*N*sizeof(float), cudaMemcpyHostToDevice);
	//matrixMul<<<Grid, Block>>>(dev_A, dev_B, dev_C, N);
	TilematrixMul<<<Grid, Block>>>(dev_A, dev_B, dev_C, N, BLOCK_SIZE);
    
	cudaMemcpy(C, dev_C, N*N*sizeof(float),cudaMemcpyDeviceToHost);
// stop timing
	cudaEventRecord( stop, 0 );

	cudaEventSynchronize( stop );
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );

	cudaEventDestroy(start); cudaEventDestroy(stop);

// resolution is 0.5 micro second, but milliseconds are reported
	printf(" %dx%d matrix mult, time %10.3e ms\n", N,N,elapsedTime);
	printf("flop rate %10.3e\n",NN*NN*NN/(elapsedTime/1000.0));

	/* open file for writing timing results                                   */
	FILE *tp = NULL;            
	tp = fopen("tile_time.csv", "w");
	fprintf(tp, "%d, %d, %10.3e, ",N, BLOCK_SIZE, elapsedTime);

	fclose(tp);



    
    free(A); free(B); free(C);
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
}

