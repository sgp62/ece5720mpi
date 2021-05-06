/*Stefen Pegels, sgp62
 /usr/local/cuda-10.1/bin/nvcc -o sgp62_hw6.out sgp62_hw6_code.cu -lcublas -lcusolver
 ./sgp62_hw6.out
 Steps to be implemented

  (1)  load data matrix A to the host.
  (2)  set a 32-length vector x to all ones.
  (3)  move A and x to the device.
  (4)  Using the library matrix-vector routine set b = Ax.
  (5)  Using the library SVD routine find the SVD of $A$, $A=U\Sigma V^T$.
  (6)  Form U_k, V_k, S_k
  (7)  Find x_k
       Using the matrix-vector multiplication routine form b_k=U_k^Tb.
       Scale elements of b_k by the corresponding diagonal elements of
       S_k^{-1}. Note that S_k^{-1} is never formed.
       Once you have the modified b_k, that is d_k = S_k^{-1}U_k^Tb_k,
       use the matrix-vector multiplication routine to form x_k = U_kd_k.
  (8)  Compute error ||x_k-x||_2 using a cuSOLVER routine.
  (9)  Compute the residual error eta_k = ||A_kx_k-b||_2 using a cuSOLVER routine.
  (10) Move x_k and eta_k to the host.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

__global__ void create_k_matrix(double * dev_M, double * dev_K, int n){

    int i = threadIdx.y; //Row i of M
    int j = threadIdx.x; //Column j of M

    dev_K[i*n + j] = dev_M[i*2*n+j]; //should create 32 x 16 
}

__global__ void create_S_inv(double * dev_S, double * dev_Si){

    int i = threadIdx.x;
    dev_Si[i] = 1 / dev_S[i];
}

__global__ void subtract(double *x, double *x_k, double * x_e){
    int i = threadIdx.x;
    x_e[i] = x_k[i] - x[i];
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    cudaError_t cudaStat1 = cudaSuccess;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    //cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);

/* (1)&(2) read A from mymatrix.txt, set x to all ones*****************
   A is 32 by 32, so set m = n = 32
   make A a 1D vector
   make sure that A is stored in the 0-index column ordering
********************************************************************/
    int i,j;
    int m,n;
  
    FILE *file;
    file=fopen("MyMatrix.txt", "r");
    //file=fopen("testMatrix.txt", "r");
  
    m = n = 32; 

    double alpha = 1.0;
    double beta = 0.0;

    double * x = (double *)malloc(n*sizeof(double));
    double * A = (double *)malloc(m*n*sizeof(double));
    double * b = (double *)malloc(n*sizeof(double));

    for(i = 0; i < m; i++){//Fill A
      for(j = 0; j < n; j++){
        if(!fscanf(file, "%lf", &A[j*m+i])) break;
      }
    }
    //printf("%.16lf, %.16lf\n",A[0], A[m*n - 1]);
    fclose(file);




    for(i = 0; i < n; i++){ //Fill x
        x[i] = 1.0;
    }

/*********************************************************************
/*   (3) move A and x to the device
********************************************************************/
    double *d_A;
    double *d_x;
    cudaMalloc (( void **)&d_A , m*n*sizeof(*A));
    cudaMalloc (( void **)&d_x ,n*sizeof(*x));

    cublasSetMatrix (m,n, sizeof (*A), A, m, d_A ,m);
    cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1);

/*********************************************************************
/*   (4) on the device call cublasDgemv to get b = A*x
y = alpha*A*x + beta*y;
cublasDgemv(handle, trans, m, n, *alpha, *A, int lda, *x, int incx,
                           *beta, *y, incy)
handle - input handle to the cuBLAS library context.
trans  - input operation op(A) is, A if CUBLAS_OP_N, A^T CUBLAS_OP_T,
         A^H if transa == CUBLAS_OP_H
m      - input number of rows of matrix A.
n      - input number of columns of matrix A.
alpha  - host or device input <type> scalar used for multiplication.
A      - device input <type> array of dimension lda x n with lda>=kl+ku+1.
lda    - input leading dimension of two-dimensional array used to store matrix A.
d_x    - device input <type> vector with n elements if transa == CUBLAS_OP_N 
         and m elements otherwise.
incx   - input stride between consecutive elements of x.
beta   - host or device input <type> scalar used for multiplication, 
         if beta == 0 then y does not have to be a valid input.
d_y    - device in/out <type> vector with m elements 
         if transa == CUBLAS_OP_N and n elements otherwise.
incy   - input stride between consecutive elements of y.


*********************************************************************************/
    double *d_b;
    cudaMalloc (( void **) &d_b ,n* sizeof(*b));
    cublasDgemv(cublasH, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_b, 1); //Creating b = Ax  ( b = 1*A*x + 0*b )

/*********************************************************************
/*   (5) on the device call cusolverDnDgesvd to get A = U*S*V^T
/*********************************************************************************
cusolverDnDgesvd(handle, jobu, jobvt, m, n, *A, lda, *S, *U, ldu, *VT, ldvt, *work,
                lwork, *rwork, *devInfo)
cusolverH - handle 
jobu      - if 'A' all columns of U are returned
jobvt     - if 'A' all rows of V^T are returned
m         - number of rows of the matrix
n         - number of columns of the matrix
d_A       - lda by n, A on the device
lda       - leading dimension of the matrix, lda = m
d_S       - vector of singuler values, min(m,n)
d_U       - lda by m, matrix of left singular vectors, lda = m
ldu       - leading dimension of d_U, set to lda = m 
d_VT      - ldvt by n matrix V^T, ldvt = lda = m, 
ldvt      - leading dimension of d_VT, set to lda 
d_work    - working space of size lwork
lwork     - size of working space returned by gesvd_bufferSize
d_rwork   - real array, contains the unconverged superdiagonal elements 
devInfo   - if devInfo = 0, the operation is successful.
*********************************************************************************/

    double * U = (double *)malloc(m*m*sizeof(double));
    double * VT = (double *)malloc(n*n*sizeof(double));
    double * S = (double *)malloc(n*sizeof(double));

    int lwork = 0;

    double *d_S;
    double *d_U;
    double *d_VT;
    int *devInfo = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;

    cudaMalloc (( void **) &d_S ,n* sizeof(*S));
    cudaMalloc (( void **) &d_U ,m*m* sizeof(*U));
    cudaMalloc (( void **) &d_VT ,n*n* sizeof(*VT));
    cudaMalloc (( void **) &devInfo ,sizeof(int));

    //Get working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    cusolver_status = cusolverDnDgesvd(cusolverH, 'A', 'A', m, n, d_A, m, d_S, d_U, m, d_VT, m, d_work,
        lwork, d_rwork, devInfo);


    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // cudaMemcpy(S, d_S, n*m*sizeof(double),cudaMemcpyDeviceToHost);
    // cudaMemcpy(U, d_U, n*m*sizeof(double),cudaMemcpyDeviceToHost);
    // cudaMemcpy(VT, d_VT, n*m*sizeof(double),cudaMemcpyDeviceToHost);

    // printf("U:\n");
    // for(int i = 0; i < n; i++){
    //     for(int j = 0; j < m; j++){
    //         printf("%5.3e ",U[j*m+i]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // printf("S:\n");
    // for(int i = 0; i < n; i++){
    //     printf("%5.3e ",S[i]);
    // }
    // printf("\n");
    // printf("\n");
    // printf("VT\n");
    // for(int i = 0; i < n; i++){
    //     for(int j = 0; j < m; j++){
    //         printf("%5.3e ",VT[j*m+i]);
    //     }
    //     printf("\n");
    // }



    //printf("%.16lf, %.16lf\n",t_S[0], t_S[n - 1]);
    //printf("%.16lf, %.16lf\n",t_U[0], t_U[m*n - 1]);
    // printf("%.16lf, %.16lf\n",t_VT[0], t_VT[m*n - 1]);




/************************************************************************/
//  (6)  Form U_k, V_k, S_k
//       U_k is 32 by 16, same with V_k, however remember that 
//       cusolverDnDgesvd returns V^T, not V
//       S_k is a vector of length 16 of the largest singular values of A     
/*********************************************************************/
    double *d_V;
    double *d_Sk;
    double *d_Uk;
    double *d_Vk;
    cudaMalloc (( void **) &d_V ,n*n* sizeof(*S));
    cudaMalloc (( void **) &d_Sk ,(n/2)* sizeof(*S));
    cudaMalloc (( void **) &d_Uk ,m*(m/2)* sizeof(*U));
    cudaMalloc (( void **) &d_Vk ,n*(n/2)* sizeof(*b));

    cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, d_VT, m, &beta, d_V, m, d_V, m); //Transposes VT and places in V on device

    cudaMemcpy(d_Sk, d_S, (n/2)*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Uk, d_U, m*(n/2)*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Vk, d_V, m*(n/2)*sizeof(double),cudaMemcpyDeviceToDevice);


    //Launch kernels to copy V, S, U values into Vk, Sk, Uk on device
    // int numBlocks = 1;
    // dim3 threadsPerBlock(m/2, m);
    // dim3 threadsPerBlockS(m/2, 1);
    // create_k_matrix<<<numBlocks,threadsPerBlock>>>(d_V, d_Vk, m/2); //32 rows, 16 col
    // cudaDeviceSynchronize();
    // create_k_matrix<<<numBlocks,threadsPerBlock>>>(d_U, d_Uk, m/2); //32 rows, 16 col
    // cudaDeviceSynchronize();
    // create_k_matrix<<<numBlocks,threadsPerBlockS>>>(d_S, d_Sk, m/2); //16 rows, 1 col
    // cudaDeviceSynchronize();

/*************************************************************
  (7)  Find x_k
       (a) use cublasDgemv to get b_k = U_k^Tb 
       (b) find a length 16 vector t with entries being inverses of
           entries of S_k 
       (c) scale b_k by t, formaly get r = diag(t)* b_k, use cublasDdgmm
           or host code

cublasDdgmm(handle, mode, m, n, *A, lda, *x, incx, *C, ldc)
C = A*diag(x) if mode == CUBLAS_SIDE_RIGHT
    diag(x)*A if mode == CUBLAS_SIDE_LEFT
cublasH  - handle 
mode     -
m        - number of rows in A = d_VT and C = d_W
n        - number of columns in A and C
A        - array A = d_VT 
lda      - leading dimension of A, lda = m
x        - vector of size m if mode == CUBLAS_SIDE_LEFT,
           of size n if mode == CUBLAS_SIDE_RIGHT, x = d_S
incx     - stride used in x, incx = 1 
C        - the result array
ldc      - Leading dimension of C, ldc = lda = m
 
       (d) x_k = V_k*r, use cublasDgemv

*********************************************************************************/

    double *d_bk;
    cudaMalloc (( void **) &d_bk ,(n/2)* sizeof(*b));

    cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T, m, n/2, &alpha, d_Uk, m, d_b, 1, &beta, d_bk, 1); //Forming d_bk
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    double *d_Si;
    cudaMalloc (( void **) &d_Si ,(n/2)* sizeof(*S));

    create_S_inv<<<1,m/2>>>(d_Sk, d_Si); //Forming d_Si, inverse of d_Sk
    cudaDeviceSynchronize();

    double *d_r;
    cudaMalloc (( void **) &d_r ,(n/2)* sizeof(*b));

    cublas_status =  cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT, m/2, 1, d_bk, m/2, d_Si, 1, d_r, m/2); //Creating d_r = diag(d_Si)*d_bk
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    double *d_xk;
    cudaMalloc (( void **) &d_xk ,n* sizeof(*x));
    cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N, m, n/2, &alpha, d_Vk, m, d_r, 1, &beta, d_xk, 1); //Forming d_xk
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

/*********************************************************************
  (8)   Compute error ||x_k-x||_2 using  cublasDnrm2
        First get x_k-x
  (9)   compute eta_k = ||A_kx_k-b||_2
        use cublasDgemv to get z = A_kx_k-b then use cublasDnrm2
*********************************************************************************/

    double *d_xe;
    double norm_xe;

    cudaMalloc ((void **) &d_xe, n* sizeof(*x));
    cudaMalloc ((void **) &norm_xe,  sizeof(*x));
    subtract<<<1,m>>>(d_x, d_xk, d_xe);
    cudaDeviceSynchronize();

    cublas_status = cublasDnrm2(cublasH, n, d_xe, 1, &norm_xe);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    
    double *d_z;
    double d_eta_k;
    double *d_Ak;

    cudaMalloc ((void **) &d_z, n* sizeof(*x));
    cudaMalloc ((void **) &d_eta_k, sizeof(double));
    cudaMalloc ((void **) &d_Ak, m*n* sizeof(*A));

    //create_k_matrix<<<numBlocks,threadsPerBlock>>>(d_A, d_Ak, 16);
    //cudaDeviceSynchronize();

    //Creating Ak
    double * Ak = (double *)malloc(n*m*sizeof(double));
    double * Uk = (double *)malloc(m*(m/2)*sizeof(double));
    double * Vk = (double *)malloc(n*(n/2)*sizeof(double));
    double * VkT = (double *)malloc(n*(n/2)*sizeof(double));
    double * Sk = (double *)malloc((n/2)*sizeof(double));
    double * Skx = (double *)malloc((n/2)*(n/2)*sizeof(double));

    cudaMemcpy(Sk, d_Sk, n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Uk, d_Uk, n*(m/2)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Vk, d_Vk, n*(m/2)*sizeof(double),cudaMemcpyDeviceToHost);

    for(int i = 0; i < n/2; i++){
        for(int j = 0; j < n/2; j++){
            if(i == j){
                Skx[j*(n/2)+i] = Sk[i];
            }
            else{
                Skx[j*(n/2)+i] = 0;
            }
        }
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n/2; j++){
            VkT[n*j+i] = Vk[n*i+j];
        }
    }

    double * temp1 = (double *)malloc(n*(n/2)*sizeof(double));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n/2; j++){
            temp1[j*n+i] = 0;
            for(int k = 0; k < n/2; k++){
                temp1[j*n+i] += Uk[k*n+i] * Skx[j*(n/2)+k];
            }
        }
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n/2; j++){
            Ak[j*n+i] = 0;
            for(int k = 0; k < n; k++){
                Ak[j*n+i] += temp1[k*n+i] * VkT[j*n+k];
            }
        }
    }

    cudaMemcpy(d_Ak, Ak, m*n*sizeof(double),cudaMemcpyHostToDevice);



    double n_beta = -1.0;
    //OH said we can use A instead of Ak
    cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N, m, n/2, &alpha, d_Ak, m, d_xk, 1, &n_beta, d_b, 1); //Forming d_b = Ax_k - d_b
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    //subtract<<<1,32>>>(d_b, d_z, d_z);

    cublas_status = cublasDnrm2(cublasH, n, d_b, 1, &d_eta_k);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
/*********************************************************************************
cublasDnrm2(cublasH, lda*n, d_A, 1, &dR_fro);
cublasH - handle
N       - number of elements, N = lda*n
x       - pointer to double precision real input vector, x = d_A
incx    - storage spacing between elements of x, incx = 1
nrm     – Euclidean norm of x, nrm = &dR_fro
*********************************************************************************/

// (10) copy x_k and eta_k back to host
    double x_k = 10.0;
    double eta_k;


    cudaMemcpy(&x_k, &norm_xe, sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&eta_k, &d_eta_k, sizeof(double),cudaMemcpyDeviceToHost);

    printf("Error: %10.3e\n", norm_xe);
    printf("Residual Error: %10.3e\n", d_eta_k);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(devInfo);

    cudaFree(d_V);
    cudaFree(d_Sk);
    cudaFree(d_Uk);
    cudaFree(d_Vk);

    cudaFree(d_bk);
    cudaFree(d_Si);
    cudaFree(d_xk);
    cudaFree(d_r);

    cudaFree(d_xe);
    cudaFree(&norm_xe);

    cudaFree(d_z);
    cudaFree(&d_eta_k);
    cudaFree(d_Ak);

    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

}