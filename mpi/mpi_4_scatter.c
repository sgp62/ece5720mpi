/* root scatters sumatrices of A to all PEs */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define P_ROWS    2       /* number of rows per PE                     */
#define K_COLUMNS 2       /* number of columns per PE                  */

int main(int argc, char **argv) {

  int rank,numtasks,count, sendcount, root = 0;
  int *A;
  /* A is the entire array, A_local is the chunk per PE */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

/* set data */
  int N = K_COLUMNS*P_ROWS*numtasks;
  if (rank == root){
    A = (int *)malloc(N*sizeof(int));
    for (int i = 0; i < N; i++) 
      A[i] = i;
     
    printf("\nRoot prints data %d by %d matrix A\n",numtasks*P_ROWS,K_COLUMNS);
    for (int i = 0; i < numtasks; i++){ 
      for(int j=0;j<K_COLUMNS;j++) 
        printf("%d  ",A[i*P_ROWS+j]);
      printf("\n");
    }      
  }
  int numel = P_ROWS*K_COLUMNS;
  int * A_local = (int *)malloc(P_ROWS*K_COLUMNS*sizeof(int));

  MPI_Scatter(A,numel, MPI_INT, A_local, numel, MPI_INT, root, MPI_COMM_WORLD);
  
  printf("rank %d, got %d elements, A_local[0] =  %d\n", rank, numel, A_local[0]);

//  free((void*) A); free((void*) A_local);
  MPI_Finalize();
}

