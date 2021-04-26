/* from matlab use                                                    *
 *  save('MyMatrix.txt', 'A', '-ascii', '-double', '-tabs')           *
 *  and then use the code below to input the matrix from MyMatrix.txt *
 *                                                                    *
 *  Alternatively, you may want to create a 1D vector as it will be   *
 *  easier to use cuBLAS routines
 */

#include<stdio.h>
#include<stdlib.h>

int main() {
  int i,j;
  int m,n;

  FILE *file;
  file=fopen("MyMatrix.txt", "r");

  m = n = 32; 

  double * mat = malloc(m*n*sizeof(double));
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      if(!fscanf(file, "%lf", &mat[i*n+j])) break;
    }
  }
  printf("%.16lf, %.16lf\n",mat[0], mat[m*n - 1]);

  // double** mat = malloc(m*sizeof(double*));
  // for(i=0;i<m;++i)
  //     mat[i]=malloc(n*sizeof(double));

  // for(i = 0; i < m; i++)
  //   for(j = 0; j < n; j++) 
  //     if (!fscanf(file, "%lf", &mat[i][j])) break;
  // printf("%.16lf, %.16lf\n",mat[0][0], mat[31][31]); 
  fclose(file);
}

