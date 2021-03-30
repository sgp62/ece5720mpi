/**
 * dim[0] = 2; dim[1] = 1, count[0] = 2, count[1] = 3
 * In brief; the 2x3 subarray to send starts at [2;1] in the 4x4 full array
 **/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
int main(int argc, char* argv[]) {

    int size, my_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    if(size != 2) {
        printf("This application is meant to be run with 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Get my rank and do the corresponding job
    switch(my_rank) {
      case 0:
      {
// Declare the full array
        int full_array[4][4];
        for(int i = 0; i < 4; i++) 
           for(int j = 0; j < 4; j++)
                    full_array[i][j] = i * 4 + j;
 
// Create the subarray datatype
        MPI_Datatype subarray_type;
        int dimensions_full_array[2] = { 4, 4 };
        int dimensions_subarray[2] = { 2, 3 };
        int start_coordinates[2] = { 2, 1 };
        MPI_Type_create_subarray(2,  dimensions_full_array, dimensions_subarray, 
			start_coordinates, MPI_ORDER_C, MPI_INT, &subarray_type);
        MPI_Type_commit(&subarray_type);
 
// Send the message
        printf("MPI process %d sends:\n-  -  -  -\n-  -  -  -\n-  %d %d %d\n- %d %d %d\n", 
		my_rank, full_array[2][1], full_array[2][2], full_array[2][3], 
		full_array[3][1], full_array[3][2], full_array[3][3]);
        MPI_Send(full_array, 1, subarray_type, 1, 0, MPI_COMM_WORLD);
        break;
     }
     case 1:
     {
// Receive the message
      int received[6];
      MPI_Recv(&received, 6, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("MPI process %d receives:\n%d %d %d %d %d %d\n", my_rank, received[0], 
	    received[1], received[2], received[3], received[4], received[5]);
            break;
     }
  }
 
  MPI_Finalize();
  return EXIT_SUCCESS;
}
