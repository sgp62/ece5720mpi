//Stefen Pegels, sgp62
/* Assignment 5, part I                                        */
/* add two vectors in CUDA, 1D grid, 1D block configuration    */
/* uses CUDA pagable memory allocations                        */
/* (1) repeat for different memory allocation schemes which    */
/*     are listed at the end of the main code                  */
/* (2) to record timings use (a) clock_gettime(), (b) clock(), */
/*     (b) cudaEventElapsedTime, with resolution of 0.5 micros */
/*         but reported in milliseconds                        */
/* (3) try different sizes of blocks to see whether this       */
/*      makes any difference                                   */

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BILLION 1000000000

__global__ void vec_add (int *d_a, int *d_b, int *d_c, int size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d_c[gid] = d_a[gid] + d_b[gid];
}

int main() {
    int size = 1 << 28;                // length of vectors to be added
    int block_size = 128;              // block size, try size 256 too
    int size_int = sizeof(int) * size;

// variables to record times by different methods
    float time;
    unsigned long Ttime;
    double TTtime;
    clock_t start,end;
    struct timespec Tstart, Tend;

// start CPU timer
    clock_gettime(CLOCK_MONOTONIC, &Tstart);

// declare CUDA events
    cudaEvent_t cstart, cstop;
    cudaEventCreate(&cstart);
    cudaEventCreate(&cstop);

 //Host pagable memory allocation
    int *h_a, *h_b, *h_c;
    h_a = (int *)malloc(size_int); 
    h_b = (int *)malloc(size_int); 
    h_c = (int *)malloc(size_int);

//Host memory initialization
    for(int i = 0; i < size; i++) {
        h_a[i] = 1; h_b[i] = 2;
    }
    memset(h_c, 0, size);

// start clock() timer
    start = clock();

// start timing CUDA events from this marker point    
    cudaEventRecord(cstart,0);

//Device memory initialization
    int *d_a, *d_b, *d_c;
    cudaMalloc((int **)&d_a,size_int); 
    cudaMalloc((int **)&d_b,size_int); 
    cudaMalloc((int **)&d_c,size_int);
   
//Host to device input data transfer
    cudaMemcpy(d_a, h_a, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_int, cudaMemcpyHostToDevice);
  
//Kernel launch
    dim3 block(block_size);
    dim3 grid(size/block.x);
    vec_add <<< grid, block >>> (d_a, d_b, d_c, size);
 
// CPU waits until device is done
    cudaDeviceSynchronize();

//Device to host output data transfer
    cudaMemcpy(h_c, d_c, size_int, cudaMemcpyDeviceToHost);

    cudaEventRecord(cstop,0);
    cudaEventSynchronize(cstop);

    end = clock();

    time = ((float)(end-start))/CLOCKS_PER_SEC;
    printf("CUDA via clock(): %10.3e secs\n",time);

    cudaEventElapsedTime(&time,cstart,cstop);
    time = time/1000;
    printf("CUDA events timer: %10.3e secs\n",time);

    cudaEventDestroy(cstart); cudaEventDestroy(cstop);

    clock_gettime(CLOCK_MONOTONIC, &Tend);

    Ttime = BILLION * (Tend.tv_sec - Tstart.tv_sec) + Tend.tv_nsec - Tstart.tv_nsec;
    TTtime = (double)Ttime/1000000000.0;
    printf("CPU clock_gettime = %10.3e secs\n", TTtime);
//  printf("CPU clock_gettime = %llu secs\n",(long long unsigned int) Ttime);

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

/* 
************* pinned memory **********
// host
  int *h_a1, *h_b1, *h_c1;
  cudaMallocHost((int **)&h_a1, size);
  cudaMallocHost((int **)&h_b1, size);
  cudaMallocHost((int **)&h_c1, size);
  for (int i = 0; i < size; i++) {
    h_a1[i] = 1; h_b1[i] = 2;
  }
  memset(h_c1, 0, size);
// device
  int *d_a1, *d_b1, *d_c1;
  cudaMalloc((int **)&d_a1, size);
  cudaMalloc((int **)&d_b1, size);
  cudaMalloc((int **)&d_c1, size);
  cudaMemcpy(d_a1, h_a1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1, size, cudaMemcpyHostToDevice);

*************** Mapped memory (zero-copy memory) ************
// host
  int *h_a2, *h_b2, *h_c2;
  cudaHostAlloc((int **)&h_a2, size, cudaHostAllocMapped);
  cudaHostAlloc((int **)&h_b2, size, cudaHostAllocMapped);
  cudaHostAlloc((int **)&h_c2, size, cudaHostAllocMapped);
  for (int i = 0; i < size; i++) {
    h_a2[i] = 1; h_b2[i] = 2;
  }
  memset(h_c2, 0, size);

// device
  int *d_a2, *d_b2, *d_c2;
  cudaHostGetDevicePointer((int **)&d_a2, (int *)h_a2, 0);
  cudaHostGetDevicePointer((int **)&d_b2, (int *)h_b2, 0);
  cudaHostGetDevicePointer((int **)&d_c2, (int *)h_c2, 0);

************** Unified memory **************************
  int *a, *b, *c;

 cudaMallocManaged((int **)&a, size);
 cudaMallocManaged((int **)&b, size);
 cudaMallocManaged((int **)&c, size);

 for (int i = 0; i < size; i++) {
   a[i] = 1; b[i] = 2;
 }
 memset(c, 0, size);

*/

