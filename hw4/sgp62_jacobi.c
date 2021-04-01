//Stefen Pegels, sgp62
/* Bren-Luk permutation for one sided Jacobi iteration for finding    *
 * the SVD of an M by K_COLUMNS matrix A, M even                      *
 * there are npes Processing Elements, npes even,                     *
 * M divisible by npes and such that N/neps = P_ROWS is even          *
 * each PE stores P_ROWS rows of A in a local array A_local           *
 * PEs are arranged in 1D Cartesian topology                          *
 * PEs 0 and npes-1 are boundary PEs, all other PEs are interior PEs  *
 * in each iteration all PEs execute the following steps:             *
 * (1) consecutive (odd,even) rows in a PE are orthogonalized by      *
 *     a 2 by 2 rotation as described in Lecture 14                   *
 * (2) for interior PEs, the second last row in the ith PE is send to *
 *     (i+1)st PE and stored in its first row                         *
 * (3) for interior PEs, the second row in the ith PE is sent to      *
 *     rank i-1 PE and is stored in its last row                      *
 * (4) the leftmost and rightmost boundary PEs are special, their     *
 *     actions depend on whether they store 2 or more rows, please    *
 *     consult notes for Lecture 14                                   *
 * (5) all other rows not mentioned in steps (2)-(4) are permuted     *
 *     according to BL ordering                                       *
 * (6) steps (1)-(5) are repeated so a complete sweep is realized     *
 * (7) a stopping criterion is checked, if satisfied the iterations   *
 *     stop, otherwise they continue from step (1)                    */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define P_ROWS    2       /* number of rows per PE                     */ // M = P_ROWS * npes
#define K_COLUMNS 4       /* number of columns                         */

int main(int argc, char ** argv) {
  int rank, npes, right, left, row_size, recvd_count;
  int rc, i, j, k, N;
  int * A;

// start MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Status stats[2];
  MPI_Request reqs[2];

  MPI_Datatype row_type;

  N = P_ROWS*K_COLUMNS*npes;

/* the master generates matrix A
 * in this example rows of are
 * [0 0 ...0] for row 0
 * [1 1 ...1] for row 1, etc.
 * this done to check whether permutations are correct
 * needs to be removed when correctness is checked      */

  if(rank == 0) {
    A = (float *) malloc(N*sizeof(float));
    printf("there are %d PEs\n",npes);
    printf("size of A is %d by %d\n",N,N);
    for(i = 0; i < npes; i++){ 
      for(j=0;j<P_ROWS;j++) {
        for(k=0;k<K_COLUMNS;k++) {
          A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k] = (float)P_ROWS*i+j;
          printf("%.3f ",A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k]);
        }
        printf("\n");
      }
    printf("--------------------\n");
    }
    for(k=0;k<N;k++) printf("%.3f ",A[k]);
      printf("\n");
  }

/* Scatter the rows to npes processes */
  int num_el = N/npes;
  float * A_local = (float *) malloc(num_el*sizeof(float));
  MPI_Scatter(A, num_el, MPI_FLOAT, A_local, num_el, MPI_FLOAT, //Ask question about sending both rows
		     0, MPI_COMM_WORLD);

  
/* you may want to check here whether MPI_Scatter was correct */
  //Each PE should have P_ROWS rows, which have K_COLUMNS elements ( P_ROWS*K_COLUMNS elems per PE)
  printf("Scatter Check\n");
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < K_COLUMNS; j++){
      printf("%.3f ", A_local[i*P_ROWS + j]);
    }
    printf("\n");
  }

/* buffers for send and receive rows */
  float *l_buf_l = (float *)calloc(K_COLUMNS, sizeof(float));
  float *l_buf_r = (float *)calloc(K_COLUMNS, sizeof(float));
  float *r_buf_l = (float *)calloc(K_COLUMNS, sizeof(float));
  float *r_buf_r = (float *)calloc(K_COLUMNS, sizeof(float));

/* Create the row_type for exchanging rows among PEs */
/* The length of a row is K_COLUMNS                  */
  MPI_Type_contiguous(K_COLUMNS, MPI_FLOAT, &row_type);
  MPI_Type_commit(&row_type);
  MPI_Type_size(row_type,&row_size);

/* starting addresses for second last and last  row in A_local * //Isn't A_Local initially only two rows?
 * these locations will be updated (swapped)                   */
  int second_last = K_COLUMNS*(P_ROWS-2);
  int second = K_COLUMNS;
  int last_row = K_COLUMNS*(P_ROWS-1);

  //threshold, error, iter, MAX_ITER not defined

/* iterate until termination criteria are not met      */ //Threshold is norm of matrix * # of elemens * Precision of float or double see Ed
while((threshold < error)&&(iter < MAX_ITER)) {

  /* perform full sweep                                   */ //Any pair of rows is orthogonal(product is 0), that is equivalent to knowing A * A^T being diagonal
                                                            //Solve for cp and sp matrix, page 2 and 3(use quadratic)
  for (k=0;k<2*npes-1;k++)  {
                                                              //Swap rows with the communication ring
                                                              //Final A is already computed as the sigma * UT, 
    /* orthogonalize consecutive (odd,even) rows            */ //Helper Function?
    //Rotation
    givens_rotate(A_local);

    if ((rank>0)&&(rank<npes-1)&&(rank%2==0)){
      for(int i = 0; i < K_COLUMNS; i++){
        l_buf_r[i] = A_local[i];
        r_buf_l[i] = A_local[K_COLUMNS+i];
      }
      //send left the second row
      MPI_Send(&l_buf_r, K_COLUMNS, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);
      //receive from left the far left row
      MPI_Recv(&l_buf_l, K_COLUMNS, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);
      //send right the second to last row
      MPI_Send(&r_buf_l, K_COLUMNS, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
      //receive from right the far right row
      MPI_Recv(&r_buf_r, K_COLUMNS, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD); 
      //Last becomes second, first becomes third(second to last) [What to do for this???]
    }

    if (rank==0){
      for(int i = 0; i < K_COLUMNS; i++){//Probably incorrect. Which was A_local???
        r_buf_l[i] = A_local[K_COLUMNS+i];
      }
      //send right the second last row
      MPI_Send(&r_buf_l, K_COLUMNS, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
      //receive from right the last row
      MPI_Recv(&r_buf_r, K_COLUMNS, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD); 
      //How to handle interactions within a PE?
    }

    if (rank == npes-1){
      // receive from left to first row
      MPI_Recv(&l_buf_l, K_COLUMNS, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD); 
      //send left the second row
      MPI_Send(&l_buf_r, K_COLUMNS, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);
    }

    if ((rank>0)&&(rank<npes-1)&&(rank%2==1)){ //Same as top one??? Literally no different it it's not on the edges
    /*  receive from left to the first row  *
    *  send left second row to last        *
    *                                      *
    *  receive from right to last,         *
    *  sendright second last to first      */
    }
  
  } /* end the k loop */

/* check termination criteria */
} /* end wwhile loop          */

  free((void *) A_local);
   
  if(rank == 0)  free((void *)A);

  MPI_Finalize();
  return 0;
}

void givens_rotate(*A){
  float * temp = (float *) malloc(K_COLUMNS*P_ROWS * sizeof(float));
  float * b = (float *) calloc(4*sizeof(float));
  //float * g = (float *) calloc(4*sizeof(float));
  float * gT = (float *) calloc(4*sizeof(float));
  //Create b out of the rows of A multiplied
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < P_ROWS; j++){
      for(int k = 0; k < K_COLUMNS; k++){
        b[i*P_ROWS+j] += A[i*K_COLUMNS + k] * A[j*K_COLUMNS + k];
      }
    }
  }

  //logic to find c and s
  float tau = (b[3] - b[0]) / (b[1]+b[2]);
  float t1 = -tau + sqrt(1 + tau * tau);
  float t2 = -tau - sqrt(1 + tau * tau);
  float t = (abs(t1) > abs(t2)) ? t1 : t2;

  //Assign c and s
  float c = 1 / sqrt(1 + t*t);
  float s = c * t;

  //Assign all 4 entries of g and gT
  // g[0] = c;
  // g[1] = s;
  // g[2] = -s;
  // g[3] = c;

  gT[0] = c;
  gT[1] = -s;
  gT[2] = s;
  gT[3] = c;

  //Copy A to temp, zero out A
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < K_COLUMNS; j++){
      temp[i*K_COLUMNS+j] = A[i*K_COLUMNS + j];
      A[i*K_COLUMNS+j] = 0.0;
    }
  }

  //Perform A = gT * temp
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < K_COLUMNS; j++){
      for(int k = 0; k < P_ROWS; k++){
        A[i*K_COLUMNS +j] += gT[i*P_ROWS + k] * temp[k*K_COLUMNS + j];
      }
    }
  }
}

