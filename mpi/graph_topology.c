/* run on 5 PEs (5 nodes)
 * there are 10 edges hardwired
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[] ) {
  int i, j, k, neighbourNumber;
  int wsize = 5;         // wsize is the number of processors
  int topo_type, my_rank;
  int nnodes, nedges;
  int *index, *edges, *outindex, *outedges, *neighbours;

  MPI_Comm comm1, comm2;
  MPI_Init( &argc, &argv ); 
  MPI_Comm_size( MPI_COMM_WORLD, &wsize ); 
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank); 

// If Processor number is more than 3 we can make a graph.
  if (wsize >= 3) { 
    index = (int*)malloc(wsize * sizeof(int) );
    edges = (int*)malloc(wsize * 2 * sizeof(int) );

// fill index values of the graph
    index[0] =2; index[1]=5; index[2]=6; index[3]=8; index[4]=10;

// fill edge values of the graph
    edges[0]=1; edges[1]=4; edges[2]=0; edges[3]=2; edges[4]=3; edges[5]=1;
    edges[6]=1; edges[7]=4; edges[8]=0; edges[9]=3;

// creat graph
  MPI_Graph_create( MPI_COMM_WORLD, wsize, index, edges, 0, &comm1 );
// MPI_COMM_WORLD communicator
// wsize   number of nodes
// index   array of node degrees 
// edges   array of graph edges
// 0       don’t to order processes in the group
// comm1   communicator which represents the graph.

// duplicate the graph and get the type
  MPI_Comm_dup( comm1, &comm2 );
  MPI_Topo_test( comm2, &topo_type );
  if(my_rank == 0)
    printf( "topology of graph is %d\n" , topo_type); 

  MPI_Graphdims_get( comm2, &nnodes, &nedges );
  if (my_rank == 0) 
    printf( "Nnodes = %d, Nedges = %d\n", nnodes,nedges);

//allocate memory for arrays
    outindex = (int*)malloc(wsize * sizeof(int) ); 
    outedges = (int*)malloc(wsize * 2 * sizeof(int)); 
    MPI_Graph_get( comm2, wsize, 2*wsize, outindex, outedges );
// wsize - length of vector outindex
// 2*wsize - length of vector outedges
// outindex - vector of integers containing degrees of nodes
// outedges - vector of integers containing the link info 
    
  if (my_rank == 0){ 
    printf( "\nnode count obtained by MPI_Graphdims_get : " );
    printf("%d",nnodes);
    printf( "\nedge count obtained by MPI_Graphdims_get : " );
    printf("%d",nedges);
    printf( "\n-------------------------------------\n");
    printf( "Array of indices obtained by MPI_Graph_get : " );
    for (i=0;i<wsize;i++) {
      printf( "%d ,", outindex[i] );
    }
    printf( "\nArray of Edges obtained by MPI_Graph_get :" );
    for ( i=0;i<2*wsize;i++) {
      printf( "%d ,", outedges[i] );
    }
    free( outindex ); free( outedges ); 
    printf( "\n-------------------------------------\n");

// print each node and its neighbours 
  if(my_rank == 0) printf("from index info\n");
    for(i=0;i<wsize;i++) {
      int temp;
      if(i==0)
        temp=0;
      else
        temp=index[i-1];

//Get each node’s neighbour number.
      neighbourNumber=index[i]-temp; 
      printf( "node no %d have %d neighbours\n", i,neighbourNumber);
      printf( "My neighbours are : ");
      for( j=temp; j<index[i];j++) {
        printf("%d,",edges[j]);
      }
      printf("\n");
    }
    printf( "\n-------------------------------------\n");
  }

//k is the node number.
  if(my_rank == 0)
    printf("from MPI_Graph_neighbors_count and MPI_Graph_neighbors\n");
    for( k=0;k<wsize;k++) {
      MPI_Graph_neighbors_count(comm2,k,&neighbourNumber);
//comm2 is the communicator we get graph’s info,  k is the node number.
// neighbourNumber is number of neighbour of “k”;
      MPI_Graph_neighbors(comm2,k,neighbourNumber,neighbours);
// neighbour is the array neighbours of k will be write.
    if(my_rank == k){
      printf( "node no %d have %d neighbours\n", k,neighbourNumber);
      printf( "My neıghbours are : ");
      for(i=0;i<neighbourNumber;i++) {
        printf("%d,",neighbours[i]);
      }
      printf("\n");
    }
    }
    free( index ); free( edges ); 
    MPI_Comm_free( &comm2 ); MPI_Comm_free( &comm1 );
 }
 MPI_Finalize(); 
 return 0;
}

