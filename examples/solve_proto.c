#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke.h>

#include "../matrices.h"

/*
  gcc -Wall -march=native  -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store  -o solve_proto solve_proto.c haar2.c matrices.c  -llapacke -lopenblas -lm
*/

#define DEFAULTM 16
#define DEFAULTN 16

#define DHAMT dhamt2_initial
#define DHIMT dhimt2_initial

void dscal_( int* N, double* DA, double* A, int* INCX );

void DHAMT( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void DHIMT( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );

void dhamt( double* restrict A, double* restrict B, int N ) {
    int i;
    
    for( i = 0 ; i < N / 2 ; i++ ){ 
        B[ i] = ( A[2*i] + A[2*i+1] ) / 2.0;
        }
    for( i = 0 ; i < N / 2 ; i++ ){ 
        B[ i + N/2] = ( A[2*i] - A[ 2*i+1] ) / 2.0;
    }    
}

void dhimt( double* restrict A, double* restrict B, int N ) {
    int i;
    
    for( i = 0 ; i < N / 2 ; i++ ) {
        B[ 2*i] = ( A[i] + A[ N / 2 + i] );
        B[2*i + 1] = ( A[i] - A[N / 2 + i] );
    }    
}

int main( int argc, char** argv ){

    int M, N;
    double* A;
    double* HA;
    double* B;
    double* HB;
    double* work;
    int *IPIV;
    double DEUX = 2.0;
    double DEMI = .5;
    int ONE = 1;
    
    if( argc < 3 ) {
        M = DEFAULTM;
        N = DEFAULTN;
    } else {
        M = atoi( argv[1] );
        N = atoi( argv[2] );
    }
    A  = (double*) malloc( M*N*sizeof( double));
    HA = (double*) malloc( M*N*sizeof( double));
    B  = (double*) malloc( M*sizeof( double));
    HB = (double*) malloc( M*sizeof( double));
    work = (double*) malloc( M*N*sizeof( double));  /* Necessaire ? */
    IPIV = (int*) malloc( M*sizeof( int));
    memset( work, 0, M*N*sizeof( double ) );
    memset( IPIV, 0, N*sizeof( int ) );

    /* Generate two random matrices */
    
    drandom( A, M, N );
    drandom( B, M, 1 );

    printf( "A:\n" );
    printmatrix( A, M, N );
    printf( "B:\n" );
    printmatrix( B, N, 1 );
    printf( "\n" );

    /* Take their Haar transform */
    
    DHAMT( A, HA, work, M, N, N, N );
    dhamt( B, HB, M  );

    printf( "HA:\n" );
    printmatrix( HA, M, N );
    printf( "HB:\n" );
    printmatrix( HB, N, 1 );
    printf( "\n" );

    /* Solve:
       AX = B
       HA HX = HB */
    
    LAPACKE_dgesv( LAPACK_ROW_MAJOR, N, 1, A, N, IPIV, B, 1 );
    LAPACKE_dgesv( LAPACK_ROW_MAJOR, N, 1, HA, N, IPIV, HB, 1 );

    printf( "X:\n" );
    printmatrix( B, N, 1 );
    printf( "HX:\n" );
    printmatrix( HB, N, 1 );
    printf( "\n" );

    /* Take the Haar transform of the solution X */
    
    /* dhamt( B, HB, M  );
    dscal_( &N, &DEUX, HB, &ONE );
    printf( "HX:\n" );
    printmatrix( HB, N, 1 ); */
    
    /* Take the inverse Haar transform of the solution HX */

    dhimt( HB, B, N );
    dscal_( &N, &DEMI, B, &ONE );
    printf( "X:\n" );
    printmatrix( B, N, 1 );

    free( A );
    free( HA );
    free( B );
    free( HB );
    free( work );
    free( IPIV );
    return EXIT_SUCCESS;
}

