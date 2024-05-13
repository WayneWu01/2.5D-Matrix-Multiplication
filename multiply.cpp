#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdlib.h> 
#include <string.h>

//create the random matrix with drand48
void generate(int n, double* &A) {
    A = (double *) malloc(n * n * sizeof(double));
    srand48(time(NULL));
    for(int i = 0; i < n * n; i++) {
        A[i] = drand48() * 10; 
    }
}
//serial multiplication for test
void serial_matrix_multiplication(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
//restore matrices for verify
void restore(double *A, double *B, double *C, double *&oA, double *&oB, double *&oC, int N, int P) {
    oA = (double*)malloc(N * N * sizeof(double));
    oB = (double*)malloc(N * N * sizeof(double));
    oC = (double*)malloc(N * N * sizeof(double));
    memset(oA, 0, N * N * sizeof(double));
    memset(oB, 0, N * N * sizeof(double));
    memset(oC, 0, N * N * sizeof(double));
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // get number blocks
    int block = N / P;
    int sq = block * block;
    for (int proc = 0; proc < size; proc++) {
        int row = proc / P;
        int col = proc % P;
        // matrix in each block
        for (int i = 0; i < block; i++) {
            for (int j = 0; j < block; j++) {
                int crow = row * block + i;
                int ccol = col * block + j;
                int ind = crow * N + ccol;
                int mat = proc * sq + i * block + j;
                // restore matrices
                oA[ind] = A[mat];
                oB[ind] = B[mat];
                oC[ind] = C[mat];
            }
        }
    }
}
int main(int argc, char **argv){
    MPI_Init(NULL, NULL);
    // get and calculate setups
    int N = atoi(argv[1]); 
    int c = atoi(argv[2]); 
    int verify = 1;
    double *A, *B, *C;
    int rank,size,coord[3],ra,rb,sa,sb;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm cart,xycart,zcart;
    MPI_Status stA,stB;
    int block = N / sqrt(size / c);
    int rs = (N / block) / c;
    int sq = block * block;
    int dims[3] = {N / block, N / block, c};
    int periods[3] = {1, 1, 0};
    if (rank == 0){
        generate(N, A);
        generate(N, B);
        //double *CC
        //CC = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        //run serial if needed
        //serial_matrix_multiplication(A, B, CC, N);
    }
    // Initailize temperary array
    double *tempA = (double *)malloc(sq * sizeof(double));
    double *tempB = (double *)malloc(sq * sizeof(double));
    double *tempC = (double *)malloc(sq * sizeof(double));
    double *resc = (double *)malloc(sq * sizeof(double));
    memset(tempA, 0, sq  * sizeof(double));
    memset(tempB, 0, sq  * sizeof(double));
    memset(tempC, 0, sq  * sizeof(double));
    memset(resc, 0, sq  * sizeof(double));
    // set mpi with dimension 3
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
    MPI_Cart_coords(cart, rank, 3, coord);
    //start mpi
    double start = MPI_Wtime();
    MPI_Cart_sub(cart, new int[3]{1, 1, 0}, &xycart);
    if (coord[2] == 0){
        MPI_Scatter(A, sq, MPI_DOUBLE, tempA, sq, MPI_DOUBLE, 0, xycart);
        MPI_Scatter(B, sq, MPI_DOUBLE, tempB, sq, MPI_DOUBLE, 0, xycart);
    }
    // steps for broadcasting A and B
    MPI_Cart_sub(cart, new int[3]{0, 0, 1}, &zcart);
    MPI_Bcast(tempA, sq, MPI_DOUBLE, 0, zcart);
    MPI_Bcast(tempB, sq, MPI_DOUBLE, 0, zcart);
    int s = (coord[1] - coord[0] + coord[2] * rs) % (N / block);
    int s1 = (coord[0] - coord[1] + coord[2] * rs) % (N / block);
    int r = (coord[1] + coord[0] - coord[2] * rs) % (N / block);
    MPI_Cart_rank(cart, new int[3]{coord[0], r, coord[2]}, &ra);
    MPI_Cart_rank(cart, new int[3]{r, coord[1], coord[2]}, &rb);
    MPI_Cart_rank(cart, new int[3]{coord[0], s, coord[2]}, &sa);
    MPI_Cart_rank(cart, new int[3]{s1, coord[1], coord[2]}, &sb);
    // send and receive blocks from mpi
    MPI_Sendrecv_replace(tempA, sq, MPI_DOUBLE, sa, 1, ra, 1, cart, &stA);
    MPI_Sendrecv_replace(tempB, sq, MPI_DOUBLE, sb, 1, rb, 1, cart, &stB);
    // block multiplication
    for (int k = 0; k < block; ++k) {
        for (int i = 0; i < block; ++i) {
            double a_ik = tempA[i * block + k];
            for (int j = 0; j < block; ++j) {
                tempC[i * block + j] += a_ik * tempB[k * block + j];
            }
        }
    }
    // Cannon's Algorithm
    MPI_Cart_shift(cart, 1, 1, &ra, &sa);
    MPI_Cart_shift(cart, 0, 1, &rb, &sb);
    for (int i = 0; i < rs - 1; i++){
        MPI_Sendrecv_replace(tempA, sq, MPI_DOUBLE, sa, 1, ra, 1, cart, &stA);
        MPI_Sendrecv_replace(tempB, sq, MPI_DOUBLE, sb, 1, rb, 1, cart, &stB);
        for (int k = 0; k < block; ++k) {
            for (int i = 0; i < block; ++i) {
                double a_ik = tempA[i * block + k];
                for (int j = 0; j < block; ++j) {
                    tempC[i * block + j] += a_ik * tempB[k * block + j];
                }
            }
        }
    }
    // gather result
    MPI_Reduce(tempC, resc, sq, MPI_DOUBLE, MPI_SUM, 0, zcart);
    if (coord[2] == 0){
        MPI_Gather(resc, sq, MPI_DOUBLE, C, sq, MPI_DOUBLE, 0, xycart);
    }
    if (rank == 0){
        double end = MPI_Wtime();
        printf("Time: %f seconds\n", end - start);
    }
    // if (rank == 0){
    //     double *oA, *oB, *oC;
    //     restore(A, B, C, oA, oB, oC, N, (int)sqrt(size/c));
    //     print(A)
    // }
    if (rank == 0){
        // if(verify){
        //     double *oA, *oB, *oC;
        //     restore(A, B, C, oA, oB, oC, N, (int)sqrt(size/c));
        //     double res = 0.0;
        //     for (int i = 0; i < N; i++) {
        //         for (int j = 0; j < N; j++) {
        //             for (int k = 0; k < N; k++) {
        //                 oC[i * N + j] -= oA[i * N + k] * oB[k * N + j];
        //             }
        //             double diff = oC[i * N + j];
        //             res += sqrt(diff * diff);
        //         }
        //     }
        //     printf("Norm: %.e\n", res);
        //     free(oA);
        //     free(oB);
        //     free(oC);
        // }
        free(A);
        free(B);
        free(C);
        free(tempA);
        free(tempB);
        free(tempC);
        free(resc);    
    }
    
    MPI_Finalize();
    return 0;
}
