#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define P_ROWS 2
#define K_COLUMNS 4

int main(int argc, char ** argv) {

    float * A = (float *) malloc(K_COLUMNS*P_ROWS * sizeof(float));
    float * temp = (float *) malloc(K_COLUMNS*P_ROWS * sizeof(float));
    float * b = (float *) calloc(4, sizeof(float));
    float * gT = (float *) calloc(4, sizeof(float));
    float * g = (float *) calloc(4, sizeof(float));
    float * d = (float *) calloc(4, sizeof(float));
    float * d2 = (float *) calloc(4, sizeof(float));

    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < K_COLUMNS; j++){
            A[i*K_COLUMNS+j] = (float) i+j;
            //printf("%.3f ", A[i*P_ROWS+j]);
        }
        //printf("\n");
    }

    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < K_COLUMNS; j++){
            printf("%.3f ", A[i*K_COLUMNS+j]);
        }
        printf("\n");
    }
    printf("\n");


    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < P_ROWS; j++){
            for(int k = 0; k < K_COLUMNS; k++){
                b[i*P_ROWS+j] += A[i*K_COLUMNS + k] * A[j*K_COLUMNS + k];
            }
        }
    }

        //logic to find c and s
    float tau = (b[3] - b[0]) / (b[1]+b[2]);
    float t1 = -tau + sqrt(1 + tau * tau);
    float t2 = -tau - sqrt(1 + tau * tau);
    float t = (abs(t1) > abs(t2)) ? t1 : t2;

    //Assign c and s
    float c = 1 / sqrt(1 + t*t);
    float s = c * t;

    //Assign all 4 entries of g and gT
    g[0] = c;
    g[1] = s;
    g[2] = -s;
    g[3] = c;

    gT[0] = c;
    gT[1] = -s;
    gT[2] = s;
    gT[3] = c;

    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < P_ROWS; j++){
            printf("%.3f ", gT[i*P_ROWS +j]);
        }
        printf("\n");
    }
    printf("\n");

    //Test do gT * B * g
    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < P_ROWS; j++){
            for(int k = 0; k < P_ROWS; k++){
                d[i*P_ROWS+j] += gT[i*P_ROWS+k] * b[k*P_ROWS+j];
            }
        }
    }
    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < P_ROWS; j++){
            for(int k = 0; k < P_ROWS; k++){
                d2[i*P_ROWS+j] += d[i*P_ROWS+k] * g[k*P_ROWS+j];
            }
        }
    }
    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < P_ROWS; j++){
            printf("%.3f ", d2[i*P_ROWS+j]);
        }
        printf("\n");
    }
    printf("\n");

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

    for(int i = 0; i < P_ROWS; i++){
        for(int j = 0; j < K_COLUMNS; j++){
            printf("%.3f ", A[i*K_COLUMNS +j]);
        }
        printf("\n");
    }

    free(A);
    free(b);
    free(gT);
    return(0);
}