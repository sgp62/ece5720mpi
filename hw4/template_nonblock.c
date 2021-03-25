/********************************************************************
 * Point-to-point nonblocking bandwidth benchmark                   *
 * Adopted from llnl.gov/tutorials                                  *
 * there must be an even number of PEs                              *
 * PEs are divided into two equal size sets, one from PE 0 to       *
 * n_size/2 - 1, the other from n_size/2 to n_size - 1              *
 * PEs from one set send messages of increasing length to PEs in    *
 * other set, and then receive messages fron the other set          *
 * the round trip time is recorded                                  *
 * this ping-pong is repeated several times and timing is averaged  *
 * graph the timings against the (log of) message length            *
 ********************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define MAXTASKS      144       /* max number of nodes in the cluster   */
#define STARTSIZE     1         /* start by sending one element         */
#define ENDSIZE       1000000   /* finish by sending ENDSIZE elements   */
#define MULT          10        /* next message size multiplied by MULT */  
#define REPETITIONS   20        /* repeat 20 times for each length      */

int main (int argc, char *argv[])
{
/* declear parameters, some are already used below in the template       */
  MPI_Status status, stats[2];
  MPI_Request reqs[2];

/***************************** initialization *****************************/ 
  int my_rank, n_tasks;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_tasks);

  start = STARTSIZE; end = ENDSIZE; mult = MULT; repet = REPETITIONS;

/* open file for writing timing results                                   */

/* fill-in the message buffer "msgbuf" of length MAXLENGTH, progressively *
 * longer parts of the buffer will be send/recived by pair of PEs         */

/* get the processor name and send it to the master, remember that message *
 * from PE i is stored at position i in the receive buffor hostmap         */
  MPI_Get_processor_name(host, &namelength);
  MPI_Gather(&host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, &hostmap,
          MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

/* Establish send/receive partners and communicate to task 0  *
 * pair (src,dest) are either (my_rank,my_rank+n_tasks/2) or  *
 * (my_rank,my_rank-n_tasks/2)                                *
 * task pairs are transmitted to the master using MPI_Gather  */

/* Report the set-up */
if (my_rank == 0) {
  resolution = MPI_Wtick();
  printf("\n******************** MPI Bandwidth Test ********************\n");
  printf("Message start size= %d bytes\n",start);
  printf("Message finish size= %d bytes\n",end);
  printf("Incremented by %d bytes per iteration\n",mult);
  printf("Roundtrips per iteration= %d\n",repet);
  printf("MPI_Wtick resolution = %e\n",resolution);
  printf("************************************************************\n");
  for (i=0; i<n_tasks; i++)
    printf("task %3d is on %s partners with %3d\n",i,hostmap[i],taskpairs[i]);
  printf("************************************************************\n");
  }

/*************************** first group of tasks *************************
 * The first group use nonblocking send/receive to communicate with their *
 * partners, calculate the bandwidth for each message size and report to  *
 * to the master timing per byte transmitted.                             * 
 * **************************************************************************/

  if (my_rank < n_tasks/2) {
    for (n = start; n <= end; n = n*mult) {
      n_bytes =  sizeof(char) * n;
      for (i = 1; i <= repet; i++){
// start timer
        MPI_Isend(&msgbuf, .......);
        MPI_Irecv(&msgbuf, .......); 
        MPI_Waitall(2, reqs, stats);
// stop timer
      }
/* tasks send their timings to task 0 */
    }
}

/**************************** second half of tasks **************************
/* The second group use nonblocking receive/send to communicate with  their *
 * partners tasks, timing is taken by the first group                       */

  if (my_rank >= n_tasks/2) {
    for (n = start; n <= end; n = n*mult) {
      for (i=1; i<=repet; i++){
        MPI_Irecv(&msgbuf, .......);
        MPI_Isend(&msgbuf, .......);
        MPI_Waitall(2, reqs, stats);
      }
    }
  }

MPI_Finalize();

}  /* end of main */

