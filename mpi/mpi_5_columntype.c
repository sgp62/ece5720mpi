/* transposes a 4 by 4 matrix
 * uses the columntype type 
 * run on 4 PEs           
 * mpirun -np 4 -hostfile my_hostfile ./a.out --mca opal_warn_on_missing_libcuda 0 */

#include <stdio.h>      
#include <stdlib.h>    
#include <string.h>
#include <math.h>
#include <mpi.h>

int main (int argc, char *argv[]){
  int npes, rank, source = 0, dest, tag = 1, i, j;
  float b[4];

  MPI_Status stat;
  MPI_Datatype columntype;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  MPI_Type_vector(4, 1, 4, MPI_FLOAT, &columntype);
  MPI_Type_commit(&columntype);

  if (npes == 4) {
    if (rank == 0) {
      printf("matrix to be transposed\n");
      float a[4][4];
        for(i =0;i<4;i++){
           for(j=0;j<4;j++){
              a[i][j] = i*4+j;
	      printf("%10.3e  ", a[i][j]);
           }
           printf("\n");
        }
	printf("\n");
        for (i=0; i<npes; i++)
           MPI_Send(&a[0][i], 1, columntype, i, tag, MPI_COMM_WORLD);
    }
    MPI_Recv(b, 4, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &stat);
    for (i=0; i<4; i++) {
      printf("%10.3e  ", b[i]);
    }
    printf("\n");
    } 
    else {
      printf("Must use 4 processors. Terminating.\n");
    }
  MPI_Finalize();
}
