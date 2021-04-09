//Stefen Pegels, sgp62
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

#define MAXTASKS      12       /* max number of nodes in the cluster   */
#define STARTSIZE     1         /* start by sending one element         */
#define ENDSIZE       1000000   /* finish by sending ENDSIZE elements   */
#define MULT          10        /* next message size multiplied by MULT */  
#define REPETITIONS   20        /* repeat 20 times for each length      */
#define ITERS         7

int main (int argc, char *argv[])
{
/* declear parameters, some are already used below in the template       */
  MPI_Status status, stats[2];
  MPI_Request reqs[2];
  int src, dest;
  char host[MPI_MAX_PROCESSOR_NAME];
  int taskpairs[MAXTASKS];
  char hostmap[MAXTASKS][MPI_MAX_PROCESSOR_NAME];

  char msgbuf[ENDSIZE];
  double start_time, end_time;
  double avg_time=0;
  int tag=0;
  double avg_data[MAXTASKS*ITERS];
  double finaltiming[ITERS];


/***************************** initialization *****************************/ 
  int my_rank, n_tasks;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_tasks);
  int start = STARTSIZE;
  int end = ENDSIZE; 
  int mult = MULT; 
  int repet = REPETITIONS;

  
  int sizes[1] = {MAXTASKS};
  int subsizes[1] = {ITERS};
  int sstart[1] = {0};
  MPI_Datatype rowvec;
  MPI_Type_create_subarray(1, sizes, subsizes, sstart, MPI_ORDER_C, MPI_DOUBLE, &rowvec);

/* open file for writing timing results                                   */
  FILE *tp = NULL;            
  tp = fopen("nonblock_time.csv", "w");

/* fill-in the message buffer "msgbuf" of length MAXLENGTH, progressively *
 * longer parts of the buffer will be send/recived by pair of PEs         */
  for(int i = 0; i < ENDSIZE; i++){
    msgbuf[i] = 'a';
  }


/* get the processor name and send it to the master, remember that message *
 * from PE i is stored at position i in the receive buffor hostmap         */
  int namelength = MPI_MAX_PROCESSOR_NAME;

  MPI_Get_processor_name(host, &namelength);
  MPI_Gather(&host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, &hostmap,
          MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

/* Establish send/receive partners and communicate to task 0  *
 * pair (src,dest) are either (my_rank,my_rank+n_tas  `ks/2) or  *
 * (my_rank,my_rank-n_tasks/2)                                *
 * task pairs are transmitted to the master using MPI_Gather  */
  if(my_rank < n_tasks/2){
    src = dest = n_tasks/2 + my_rank;
  }
  if(my_rank >= n_tasks/2){
    src = dest =  my_rank - n_tasks/2;
  }
  MPI_Gather(&dest, 1, MPI_INT, &taskpairs, 1, MPI_INT, 0, MPI_COMM_WORLD); //OH

/* Report the set-up */
  if (my_rank == 0) {
    double resolution = MPI_Wtick();
    printf("\n******************** MPI Bandwidth Test ********************\n");
    printf("Message start size= %d bytes\n",start);
    printf("Message finish size= %d bytes\n",end);
    printf("Incremented by %d bytes per iteration\n",mult);
    printf("Roundtrips per iteration= %d\n",repet);
    printf("MPI_Wtick resolution = %e\n",resolution);
    printf("************************************************************\n");
    for (int i=0; i<n_tasks; i++)
      printf("task %3d is on %s partners with %3d\n",i,hostmap[i],taskpairs[i]);
    printf("************************************************************\n");
  }
/*************************** first group of tasks *************************
 * The first group use nonblocking send/receive to communicate with their *
 * partners, calculate the bandwidth for each message size and report to  *
 * to the master timing per byte transmitted.                             * 
 * **************************************************************************/
  double local_avg[ITERS];
  int k = 0;
  if (my_rank < n_tasks/2) {
    for (int n = start; n <= end; n = n*mult) {
      avg_time = 0;
      int n_bytes =  sizeof(char) * n;
      for (int i = 1; i <= repet; i++){
        // start timer
        start_time = MPI_Wtime();
        MPI_Isend(&msgbuf, n, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&msgbuf, n, MPI_CHAR, src, tag, MPI_COMM_WORLD, &reqs[1]); 
        MPI_Waitall(2, reqs, stats);
        // stop timer
        end_time = MPI_Wtime();
        avg_time += end_time - start_time;
      }
      avg_time = avg_time / repet;
      avg_time = avg_time / n;
      local_avg[k] = avg_time;
      k+=1;
/* tasks send their timings to task 0 */
    }
    //MPI_Gather(&local_avg, 1, rowvec, &avg_data, 1, rowvec, 0, MPI_COMM_WORLD);


  }
  MPI_Gather(&local_avg, ITERS, MPI_DOUBLE, &avg_data, ITERS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //if(my_rank == 0){
    //printf("First if\n");
  //MPI_Ireduce(&local_avg, &avg_data, MAXTASKS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &reqs[2]);
  //MPI_Waitall(MAXTASKS, reqs, stats);

 //}
  if(my_rank == 0){
    for(int i = 1; i <= ITERS; i++){
      finaltiming[i-1] = 0;
      for(int j = 0; j < n_tasks/2; j++){
        finaltiming[i-1] += avg_data[j*ITERS + i-1];
      }
    }
    // for(int i = 0; i < (ITERS *n_tasks/2); i++){
    //   fprintf(tp, "%1.3e\n", avg_data[i]);
    // }
    for(int i = 0; i < ITERS; i++){
      fprintf(tp, "%d, ", i);
      fprintf(tp, "%1.3e, ", finaltiming[i]);
      fprintf(tp, "\n");
    }
  }
/**************************** second half of tasks **************************
/* The second group use nonblocking receive/send to communicate with  their *
 * partners tasks, timing is taken by the first group                       */

  if (my_rank >= n_tasks/2) {
    for (int n = start; n <= end; n = n*mult) {
      int n_bytes =  sizeof(char) * n;
      for (int i=1; i<=repet; i++){
        MPI_Irecv(&msgbuf, n, MPI_CHAR, src, tag, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&msgbuf, n, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &reqs[0]);
        MPI_Waitall(2, reqs, stats);
      }
    }
  }
  
  MPI_Finalize();
  fclose(tp);


}  /* end of main */

