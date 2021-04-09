//Stefen Pegels, sgp62
/* Bren-Luk permutation for one sided Jacobi iteration for finding    *
 * the SVD of an M by K_COLUMNS matrix A, M even                      *
 * there are npes Processing Elements, npes even,                     *
 * M divisible by npes and such that N/neps = P_ROWS is even          *
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
#include <math.h>

#define P_ROWS    2       /* number of rows per PE                     */ // M = P_ROWS * npes
#define K_COLUMNS 128       /* number of columns                         */
#define MAX_SWEEPS        1024        

double calculate_error(double *A, double *AT){
  double result;
  double * temp = (double *) calloc(4, sizeof(double));

  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < P_ROWS; j++){
      for(int k = 0; k < K_COLUMNS; k++){
        temp[i*P_ROWS+j] += A[i*K_COLUMNS+k] * AT[k*P_ROWS + j];
      }
    }
  }
  result += temp[1] * temp[1] + temp[2] * temp[2]; //sum of squares of nondiagonal elements, temp is 2x2 because it is A * AT
  return result;

}
void givens_rotate(double *A){
  double * temp = (double *) malloc(K_COLUMNS*P_ROWS * sizeof(double));
  double * b = (double *) calloc(4,sizeof(double));
  //double * g = (double *) calloc(4*sizeof(double));
  double * gT = (double *) calloc(4,sizeof(double));
  //Create b out of the rows of A multiplied
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < P_ROWS; j++){
      for(int k = 0; k < K_COLUMNS; k++){
        b[i*P_ROWS+j] += A[i*K_COLUMNS + k] * A[j*K_COLUMNS + k];
      }
    }
  }

  //logic to find c and s
  double denom = b[1]+b[2];
  if(denom == 0) denom += 1e-32;
  double tau = (b[3] - b[0]) / (denom);
  double t1 = -tau + sqrt(1 + tau * tau);
  double t2 = -tau - sqrt(1 + tau * tau);
  double t = (abs(t1) > abs(t2)) ? t1 : t2;

  //Assign c and s
  double c = 1 / sqrt(1 + t*t);
  double s = c * t;

  //Assign all 4 entries of g and gT
  // g[0] = c;
  // g[1] = s;
  // g[2] = -s;
  // g[3] = c;

  gT[0] = c;
  gT[1] = -s;
  gT[2] = s;
  gT[3] = c;

  //Copy A to temp, zero out A
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < K_COLUMNS; j++){
      temp[i*K_COLUMNS+j] = A[i*K_COLUMNS + j];
      A[i*K_COLUMNS+j] = 0.0;
    }
  }

  //Perform A = gT * temp
  for(int i = 0; i < P_ROWS; i++){
    for(int j = 0; j < K_COLUMNS; j++){
      for(int k = 0; k < P_ROWS; k++){
        A[i*K_COLUMNS +j] += gT[i*P_ROWS + k] * temp[k*K_COLUMNS + j];
      }
    }
  }
}

int main(int argc, char ** argv) {
  int rank, npes, right, left, row_size, recvd_count;
  int rc, i, j, k, N;
  double * A;
  double start_time, end_time;
  
  FILE *tp = NULL;            
  tp = fopen("jacobi_time.csv", "w");

// start MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Status stats[2];
  MPI_Request reqs[2];
  MPI_Request reqs2[2];

  MPI_Datatype row_type;

  N = P_ROWS*K_COLUMNS*npes;

  if(rank == 0) {
    A = (double *) malloc(N*sizeof(double));
    
    //printf("there are %d PEs\n",npes);
    //printf("size of A is %d by %d\n",npes*P_ROWS,K_COLUMNS);
    for(i = 0; i < npes; i++){ 
      for(j=0;j<P_ROWS;j++) {
        for(k=0;k<K_COLUMNS;k++) {
          A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k] = drand48();
          //printf("%.3f ",A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k]);
        }
        //printf("\n");
      }
    //printf("--------------------\n");
    }
    // for(k=0;k<N;k++) printf("%.3f ",A[k]);
    //   printf("\n");
  }

/* Scatter the rows to npes processes */
  int num_el = N/npes;
  double * A_local = (double *) malloc(num_el*sizeof(double));
  double * AT = (double *) malloc(num_el*sizeof(double));
  MPI_Scatter(A, num_el, MPI_DOUBLE, A_local, num_el, MPI_DOUBLE, //Ask question about sending both rows
		     0, MPI_COMM_WORLD);

  
/* you may want to check here whether MPI_Scatter was correct */
  //Each PE should have P_ROWS rows, which have K_COLUMNS elements ( P_ROWS*K_COLUMNS elems per PE)
  // printf("Scatter Check\n");
  // for(int i = 0; i < P_ROWS; i++){
  //   for(int j = 0; j < K_COLUMNS; j++){
  //     printf("%.3f ", A_local[i*K_COLUMNS + j]);
  //   }
  //   printf("\n");
  // }
 
/* buffers for send and receive rows */
  double *l_buf_s = (double *)calloc(K_COLUMNS, sizeof(double));
  double *l_buf_r = (double *)calloc(K_COLUMNS, sizeof(double));
  double *r_buf_s = (double *)calloc(K_COLUMNS, sizeof(double));
  double *r_buf_r = (double *)calloc(K_COLUMNS, sizeof(double));

/* Create the row_type for exchanging rows among PEs */
/* The length of a row is K_COLUMNS                  */
  MPI_Type_contiguous(K_COLUMNS, MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);
  MPI_Type_size(row_type,&row_size);

/* starting addresses for second last and last  row in A_local * //
 * these locations will be updated (swapped)                   */
  int second_last = K_COLUMNS*(P_ROWS-2);
  //int second = K_COLUMNS;
  int last_row = K_COLUMNS*(P_ROWS-1);

  //row numbers for A_local rows (will change)
  //Each PE will get two rows in the initial scatter
  int first = rank * P_ROWS;
  int second = rank * P_ROWS + 1;
  //printf("Rank: %d\n",rank);
  

  //threshold, error, iter, MAX_ITER not defined

/* iterate until termination criteria are not met      */ //Threshold is norm of matrix * # of elemens * Precision of double or double see Ed
  double threshold, total_error, pe_error;
  int iter = 0;
  if(rank == 0){
    start_time = MPI_Wtime();
  }
  total_error = 1e-15;
  threshold = 1e-16;
  while(iter < MAX_SWEEPS){
    iter++;
    for (k=0;k<2*npes-1;k++)  {
      /* orthogonalize consecutive (odd,even) rows            */ //Helper Function?
      //Rotation
      //dummy(A_local);
      givens_rotate(A_local);

      if ((rank>0)&&(rank<npes-1)){
        //Deposit A_local into buffers to be sent
        for(int i = 0; i < K_COLUMNS; i++){
          l_buf_s[i] = A_local[i];
          r_buf_s[i] = A_local[K_COLUMNS+i];
        }
        
        //send left buffer to the right
        MPI_Isend(l_buf_s, K_COLUMNS, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[0]);
        //send right buffer to the left
        MPI_Isend(r_buf_s, K_COLUMNS, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[1]);
        //receive from left for left buffer
        MPI_Irecv(l_buf_r, K_COLUMNS, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs2[0]);
        //receive from right for right buffer
        MPI_Irecv(r_buf_r, K_COLUMNS, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs2[1]); 

        MPI_Waitall(2, reqs, stats);
        MPI_Waitall(2, reqs2, stats);
        
        //Deposit receive buffers into A_local
        for(int i = 0; i < K_COLUMNS; i++){
          A_local[i] = l_buf_r[i];
          A_local[K_COLUMNS+i] = r_buf_r[i];
        }
        //Update first row logic for A update
        if(first == 2){
          first = 1;
        }
        else if(first%2==0){
          first -= 2;
        }
        else if(first == npes*2 -1){
          first = npes*2 -2;
        }
        else if(first%2==1){
          first += 2;
        }
        //Update second row logic for A update
        if(second == 2){
          second = 1;
        }
        else if(second%2==0){
          second -= 2;
        }
        else if(second == npes*2 -1){
          second = npes*2 -2;
        }
        else if(second%2==1){
          second += 2;
        }
      }

      if (rank==0){
        //First row is always the same - always row 0
        for(int i = 0; i < K_COLUMNS; i++){
          r_buf_s[i] = A_local[K_COLUMNS+i];
        }
        //send right to the right
        MPI_Isend(r_buf_s, K_COLUMNS, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[0]);
        //receive from right 
        MPI_Irecv(r_buf_r, K_COLUMNS, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[1]); 

        MPI_Waitall(2, reqs, stats);

        for(int i = 0; i < K_COLUMNS; i++){
          A_local[K_COLUMNS+i] = r_buf_r[i];
        }
        //First is always 0
        first = 0;
        //Update second row logic for A update
        if(second == 2){
          second = 1;
        }
        else if(second%2==0){
          second -= 2;
        }
        else if(second == npes*2 -1){
          second = npes*2 -2;
        }
        else if(second%2==1){
          second += 2;
        }
      }

      if (rank == npes-1){
        for(int i = 0; i < K_COLUMNS; i++){
          l_buf_s[i] = A_local[i]; //Moves to right, held here as a temp
          r_buf_s[i] = A_local[K_COLUMNS+i];
        }
        // receive from left
        MPI_Irecv(l_buf_r, K_COLUMNS, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[0]); 
        //send right buffer to the left
        MPI_Isend(r_buf_s, K_COLUMNS, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[1]);

        MPI_Waitall(2, reqs, stats);
        for(int i = 0; i < K_COLUMNS; i++){
          A_local[i] = l_buf_r[i];
          A_local[K_COLUMNS+i] = l_buf_s[i];
        }
        //Update first row logic for A update
        if(first == 2){
          first = 1;
        }
        else if(first%2==0){
          first -= 2;
        }
        else if(first == npes*2 -1){
          first = npes*2 -2;
        }
        else if(first%2==1){
          first += 2;
        }
        //Update second row logic for A update
        if(second == 2){
          second = 1;
        }
        else if(second%2==0){
          second -= 2;
        }
        else if(second == npes*2 -1){
          second = npes*2 -2;
        }
        else if(second%2==1){
          second += 2;
        }
      }
    

    } /* end the k loop */
    if(iter > 8){
      //MPI_Gather(A_local, P_ROWS*K_COLUMNS, MPI_DOUBLE, A, P_ROWS*K_COLUMNS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      //Error checking
      //Create Transpose of A_local
      for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < K_COLUMNS; j++){
          AT[j*P_ROWS+i] = A_local[i*K_COLUMNS+j];
        }
      }
      pe_error = calculate_error(A_local, AT);
      pe_error = abs(pe_error);
      MPI_Allreduce(&pe_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Splitting error calculations among PEs
    
    }

  }/* end wwhile loop          */


  if(rank == 0){
    end_time = MPI_Wtime();
    fprintf(tp,"%d, ",K_COLUMNS);
    fprintf(tp,"%d, ",npes);
    fprintf(tp,"%1.3e, ",end_time-start_time);
  }


  // printf("A_local for rank %d: ", rank);
  // for(int i = 0; i < P_ROWS; i++){
  //   for(int j = 0; j < K_COLUMNS; j++){
  //     printf("%1.3e ", A_local[i*K_COLUMNS+j]);
  //   } 
  //   printf("\n");
  // }

  // if(rank == 0){
  //   for(int i = 0; i < 2*npes; i++){
  //     for(int j = 0; j < K_COLUMNS; j++){
  //       printf("%.3f ", A[i*K_COLUMNS +j]);
  //     }
  //     printf("\n");
  //   }
  // }


  free((void *) A_local);
   
  if(rank == 0)  free((void *)A);

  MPI_Finalize();
  return 0;
}


