//Stefen Pegels, sgp62
/* Bren-Luk permutation for one sided Jacobi iteration for finding    *
 * the SVD of an N by K_COLUMNS matrix A, N even                      *
 * there are npes Processing Elements, npes even,                     *
 * M divisible by neps and such that N/neps = P_ROWS is even          *
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

#define P_ROWS    2       /* number of rows per PE                     */
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
    A = (int *) malloc(N*sizeof(int));
    printf("there are %d PEs\n",npes);
    printf("size of A is %d by %d\n",N,N);
    for(i = 0; i < npes; i++){ 
      for(j=0;j<P_ROWS;j++) {
        for(k=0;k<K_COLUMNS;k++) {
          A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k] = P_ROWS*i+j;
          printf("%d ",A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k]);
        }
        printf("\n");
      }
    printf("--------------------\n");
    }
    for(k=0;k<N;k++) printf("%d ",A[k]);
      printf("\n");
  }

/* Scatter the rows to npes processes */
  int num_el = N/npes;
  int * A_local = (int *) malloc(num_el*sizeof(int));
  MPI_Scatter(A, num_el, MPI_INT, A_local, num_el, MPI_INT, 
		     0, MPI_COMM_WORLD);

  
/* you may want to check here whether MPI_Scatter was correct */

/* buffers for send and receive rows */
  int *l_buf_l = (int *)calloc(K_COLUMNS, sizeof(int));
  int *l_buf_r = (int *)calloc(K_COLUMNS, sizeof(int));
  int *r_buf_l = (int *)calloc(K_COLUMNS, sizeof(int));
  int *r_buf_r = (int *)calloc(K_COLUMNS, sizeof(int));

/* Create the row_type for exchanging rows among PEs */
/* The length of a row is K_COLUMNS                  */
  MPI_Type_contiguous(K_COLUMNS, MPI_INT, &row_type);
  MPI_Type_commit(&row_type);
  MPI_Type_size(row_type,&row_size);

/* starting addresses for second last and last  row in A_local *
 * these locations will be updated (swapped)                   */
  int second_last = K_COLUMNS*(P_ROWS-2);
  int second = K_COLUMNS;
  int last_row = (P_ROWS-1)*K_COLUMNS;

/* iterate until termination criteria are not met      */
while((threshold < error)&&(iter < MAX_ITER)) {

  /* perform full sweep                                   */
  for (k=0;k<2*npes-1;k++)  {
	
    /* orthogonalize consecutive (odd,even) rows            */

    if ((rank>0)&&(rank<npes-1)&&(rank%2==0)){
    /* send right second last row,        *
    * receive from right to the last row *
    *                                    *
    * send left second row               *
    * receive from left to first row     */
    }

    if (rank==0){
    // send right second last row, receive from right to last
    }
  
    if (rank == npes-1){
    // receive from left to first row, send left second to last 
    }

    if ((rank>0)&&(rank<npes-1)&&(rank%2==1)){
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
