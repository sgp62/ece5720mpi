#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
  FILE *fp = NULL;
  if (( fp = popen("gnuplot mult_plot.gp", "w")) == NULL){
      perror("popen");
      exit(1);
  }
    
  /* Close the pipe */
  pclose(fp); 
  return(0);
}