/* split cartesian topology to smaller dimensions              *
 * partition a 2 x 3 torus along its first dimension to obtain *
 * 2 rings of 3 MPI processes each                             *
 * run with 6 PEs                                              *
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != 6) {
        printf("This application is meant to be run with 6 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

// Ask MPI to decompose our processes in a 2D cartesian grid for us
    int dims[2] = {2, 3};

// Make both dimensions periodic
    int periods[2] = {1,1};

// Let MPI assign arbitrary ranks if it deems it necessary
    int reorder = 1;

    // Create a 2D torus communicator
    MPI_Comm torus_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &torus_comm);

// My rank in the new communicator
    MPI_Comm_rank(torus_comm, &my_rank);

// Get my coordinates in the new communicator
    int my_coords[2];
    MPI_Cart_coords(torus_comm, my_rank, 2, my_coords);

// Print my location in the 2D cartesian topology.
    printf("PE %d located at (%d, %d) in the 2D torus.\n", my_rank, my_coords[0], my_coords[1]);

// Partition the 2D topology along dimension 0, preserve dimension 1
    int remain_dims[2] = {0, 1};
    MPI_Comm ring_comm;
    MPI_Cart_sub(torus_comm, remain_dims, &ring_comm);

// Get the ranks of all MPI processes in my subgrid and print it
    int subgrid_ranks[3];
    MPI_Allgather(&my_rank, 1, MPI_INT, subgrid_ranks, 1, MPI_INT, ring_comm);
    printf("PE %d in the 1D subgrid that contains PEs %d, %d and %d.\n", my_rank, subgrid_ranks[0],
           subgrid_ranks[1], subgrid_ranks[2]);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
