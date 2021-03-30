#include "mpi.h"
#include <stdio.h>

int main( int argc, char *argv[] ) {
    int buf[10];
    MPI_Aint a1, a2;    //variables able to contain a memory address

    MPI_Init( &argc, &argv );

    MPI_Get_address( &buf[0], &a1 );
    MPI_Get_address( &buf[1], &a2 );

    if ((int)(a2-a1) == sizeof(int)) {
        printf( "Get_address returned the correct distance \n" );
    }
    MPI_Finalize();
    return 0;
}
