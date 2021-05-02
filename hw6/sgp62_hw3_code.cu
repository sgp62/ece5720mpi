/*
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

/* (1)&(2) read A from mymatrix.txt, set x to all ones*****************
   A is 32 by 32, so set m = n = 32
   make A a 1D vector
   make sure that A is stored in the 0-index column ordering
********************************************************************/
    int i,j;
    int m,n;
  
    FILE *file;
    file=fopen("MyMatrix.txt", "r");
  
    m = n = 32; 

    double * x = malloc(n*sizeof(double));
    double * A = malloc(m*n*sizeof(double));
    double * b = malloc(n*sizeof(double));

    for(i = 0; i < m; i++){//Fill A
      for(j = 0; j < n; j++){
        if(!fscanf(file, "%lf", &A[i*n+j])) break;
      }
    }
    //printf("%.16lf, %.16lf\n",mat[0], mat[m*n - 1]);
    fclose(file);

    for(i = 0; i < n; i++){ //Fill x
        x[i] = 1.0;
    }

/*********************************************************************
/*   (3) move A and x to the device
********************************************************************/
    void *d_A, d_x;
    cudaMalloc ((void**)&d_A , sizeof(*A)*m*n);
    cudaMalloc ((void**)&d_x ,n*sizeof(*x));

    stat = cublasSetMatrix (m,n, sizeof (*A), A, m, d_A ,m);
    stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1);

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
    void *d_b;
    cudaMalloc (( void **) &d_b ,n* sizeof(*b));
    cublasDgemv(cublasH, CUBLAS_OP_N, m, n, 1, d_A, m, d_x, 1, 0, d_b, 1); //Creating b = Ax  ( b = 1*A*x + 0*b )
   
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

    double * U = malloc(m*m*sizeof(double));
    double * VT = malloc(n*n*sizeof(double));
    double * S = malloc(n*sizeof(double));

    int lwork = 0;

    double *d_S;
    double *d_U;
    double *d_VT;
    int *devInfo = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;
    double *d_W = NULL;  // W = S*VT

    cudaMalloc (( void **) &d_S ,n* sizeof(*S));
    cudaMalloc (( void **) &d_U ,m*m* sizeof(*U));
    cudaMalloc (( void **) &d_VT ,n*n* sizeof(*b));
    cudaMalloc (( void **) &devInfo ,sizeof(int));
    cudaMalloc (( void **) &d_W ,m*n* sizeof(double));

    //Get working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolverDnDgesvd(cusolverH, 'A', 'A', m, n, d_A, m, d_S, d_U, ldu, d_VT, m, d_work,
        lwork, d_rwork, devInfo);

    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

/************************************************************************/
//  (6)  Form U_k, V_k, S_k
//       U_k is 32 by 16, same with V_k, however remember that 
//       cusolverDnDgesvd returns V^T, not V
//       S_k is a vector of length 16 of the largest singular values of A     
/*********************************************************************/
    double *d_Sk;
    double *d_Uk;
    double *d_Vk;
    cudaMalloc (( void **) &d_Sk ,(n/2)* sizeof(*S));
    cudaMalloc (( void **) &d_Uk ,m*(m/2)* sizeof(*U));
    cudaMalloc (( void **) &d_Vk ,n*(n/2)* sizeof(*b));

    

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



/*********************************************************************
  (8)   Compute error ||x_k-x||_2 using  cublasDnrm2
        First get x_k-x
  (9)   compute eta_k = ||A_kx_k-b||_2
        use cublasDgemv to get z = A_kx_k-b then use cublasDnrm2
*********************************************************************************/
    double dR_fro = 0.0;
    cublas_status = cublasDnrm2(cublasH, n, d_A, 1, &dR_fro);
/*********************************************************************************
cublasDnrm2(cublasH, lda*n, d_A, 1, &dR_fro);
cublasH - handle
N       - number of elements, N = lda*n
x       - pointer to double precision real input vector, x = d_A
incx    - storage spacing between elements of x, incx = 1
nrm     – Euclidean norm of x, nrm = &dR_fro
*********************************************************************************/

// (10) copy x_k and eta_k back to host
}