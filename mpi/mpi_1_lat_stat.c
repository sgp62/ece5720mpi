
// From https://computing.llnl.gov/tutorials/mpi/
// mpirun -np 2 -hostfile my_hostfile ./a.out --mca opal_warn_on_missing_libcuda 0

#include "mpi.h"
#include <stdio.h>
#define	NUMBER_REPS	100
#define LENGTH 2<<10


int main(int argc, char *argv[])  {
int n, numtasks, rank, dest, source, rc, count, tag=1;  

char inmsg[LENGTH], outmsg[LENGTH];
double T1, T2, deltaT, res, avgT;              /* start/end times per rep */
MPI_Status Stat;   // required variable for receive routines

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/* task 0 sends to task 1 and waits to receive a return message */
if (rank == 0) {
  dest = 1;
  source = 1;
  res = MPI_Wtick();
  printf("clock resolution in %e\n",res);

  T1 = MPI_Wtime();     /* start time */

  for (n = 1; n <= NUMBER_REPS; n++){
    MPI_Send(outmsg, LENGTH, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    MPI_Recv(inmsg, LENGTH, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
  }
  T2 = MPI_Wtime();     /* end time */
  deltaT = T2 - T1;
  avgT= (deltaT*1000000)/(2*NUMBER_REPS); /* divide by 2 to get one way time */
  printf("the one way average latency time is %2.8f micro-sec\n", avgT);
} 
else if (rank == 1) { // task 1 waits for task 0 message then returns a message
  dest = 0;
  source = 0;
  for (n = 1; n <= NUMBER_REPS; n++){
    MPI_Recv(inmsg, LENGTH, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
    MPI_Send(outmsg, LENGTH, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
  }
}

MPI_Get_count(&Stat, MPI_CHAR, &count);
printf("Task %d: Received %d char(s) from task %d with tag %d \n",
			rank, count, Stat.MPI_SOURCE, Stat.MPI_TAG);

MPI_Finalize();
}
