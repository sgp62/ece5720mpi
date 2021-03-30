/*     2 PEs only
 *     3 by 3 array a = {0,1,...,7,8}
 *     send 
 *     block 0: element 0,      displacement 0
 *     block 1: element 3,4,    displacement 3
 *     block 2: element 6,7,8   displacement 6
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {

    int size, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

// check only 2 processes are used
    if(size != 2) {
        printf("This application is meant to be run with 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    switch(my_rank) {
      case 0: {
// Create the datatype
        MPI_Datatype triangle_type;
        int lengths[3] = { 1, 2, 3 };
        int displacements[3] = { 0, 3, 6 };
        MPI_Type_indexed(3, lengths, displacements, MPI_INT, &triangle_type);
        MPI_Type_commit(&triangle_type);

// Send the message
        int buffer[3][3] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        MPI_Request request;
        printf("MPI process %d sends values:\n%d\n%d %d\n%d %d %d\n", 
		my_rank, buffer[0][0], buffer[1][0], buffer[1][1], buffer[2][0], 
		buffer[2][1], buffer[2][2]);
        MPI_Send(buffer, 1, triangle_type, 1, 0, MPI_COMM_WORLD);
        break;
        }
     case 1: {
// Receive the message
       int received[6];
       MPI_Recv(&received, 6, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       printf("MPI process %d received values:\n%d\n%d %d\n%d %d %d\n", 
	       my_rank, received[0], received[1], received[2], received[3], 
	       received[4], received[5]);
         break;
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
