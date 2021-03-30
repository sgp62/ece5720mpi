// mpirun -np 8 -hostfile my_hostfile filename --mca opal_warn_on_missing_libcuda 0

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int myid, numprocs, left, right, count;
    char buffer[10], buffer2[10];
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    char result[50];
    int num = myid + 10;
    sprintf(result, "%d", num);

    right = (myid + 1) % numprocs;
    left = myid - 1;
    if (left < 0)
        left = numprocs - 1;

    MPI_Sendrecv(buffer, 10, MPI_CHAR, left, 123, buffer2, 10, MPI_CHAR, right, 123, MPI_COMM_WORLD, &status);
    printf("got from %d, send to %d\n",left,right);

    MPI_Get_count(&status, MPI_CHAR, &count);
printf("Task %d: Received %d char(s) from task %d with tag %d \n",
                        myid, count, status.MPI_SOURCE, status.MPI_TAG);

    MPI_Finalize();
    return 0;
}
