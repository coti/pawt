#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <lapacke.h>

#define DEFAULTM 16
#define DEFAULTN 16

void ddiago( double* A, int M, int N, double cond );
void dillrandom( double* A, int M, int N, int lda, double cond, double* work, double* work2 );
void dhamt2_( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* ); 

void printmatrix( double* mat, int M, int N ) {
    int i, j;
    printf( " - - - - - - - - - -\n" );
    for( i = 0 ; i < M ; i++ ) {
        for( j = 0 ; j < N ; j++ ) {
            printf( "%lf  ", mat[N*i+j] );
        }
        printf( "\n" );
    }
}

void printmatrixOctave( double* mat, int M, int N ) {
    int i, j;
    printf( "[" );
    for( i = 0 ; i < M ; i++ ) {
        printf( "[" );
        for( j = 0 ; j < N ; j++ ) {
            printf( "%lf, ", mat[N*i+j] );
        }
        printf( "]\n" );
    }
        printf( "]\n" );
}

int main( int argc, char** argv ){

    int M, N;
    double* mat;
    double* work;
    double* wor2;
    double cond = 1e10;

    if( argc < 3 ) {
        M = DEFAULTM;
        N = DEFAULTN;
    } else {
        M = atoi( argv[1] );
        N = atoi( argv[2] );
    }
    mat  = (double*) malloc( M*N*sizeof( double));
    work = (double*) malloc( M*N*sizeof( double));
    wor2 = (double*) malloc( M*N*sizeof( double));
    memset( wor2, 0, M*N*sizeof( double ) );

    /* Ill-conditionned matrix */
    
    dillrandom( mat, M, N, N, cond, work, wor2 );
    //    printmatrixOctave( mat, M, N );
    printmatrix( mat, M, N );

    /* Haar transform */

    dhamt2_( mat, work, wor2, M, N, N, N );
    //    printmatrixOctave( work, M, N );
    printmatrix( work, M, N );

    free( mat );
    free( work );
    free( wor2 );
    return EXIT_SUCCESS;
}

void dillrandom( double* A, int M, int N, int lda, double cond, double* work, double* work2 ) {

    int ISEED[4] = {0,0,0,1};
    double* tau;
    int i, j;
    char Transpose = 'T', NoTranspose = 'N';
    double c__1 = 1.0;

    tau = (double*) malloc( MIN( M, N) * sizeof( double ));
    
    /* fill a diagonal matrix */

    memset( A, 0, M*N*sizeof( double ) );
    ddiago( A, M, N, cond );

    /* Multiply it by random orthogonal matrices, on both sides */
    /* could we use dlagsy?*/

    /* Random M*N orthogonal matrix */
    
    LAPACKE_dlarnv( 1, ISEED, M*N, work );
    LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, N, M, work, lda, tau );

    for( i = 0 ; i < MIN( M-1, N-1 ) ; i++ ) {
        for( j = i+1 ; j < N ; j++ ) {
            work[i*N+j] = 0;
        }
    }
    for( i = 0 ; i < MIN( M, N ) ; i++ ){
        work[i*(N+1)] = 1;
    }

    /* R*A: work2 <- work * diag */
    
    dgemm_( &NoTranspose, &NoTranspose, &M, &N, &N, &c__1,
           work, &N, A, &N, &c__1, work2, &N ); 

    /* Random N*N orthogonal matrix */
    
    LAPACKE_dlarnv( 1, ISEED, N*N, work );
    LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, N, N, work, N, tau );

    for( i = 0 ; i < N-1 ; i++ ) {
        for( j = i+1 ; j < N ; j++ ) {
            work[i*N+j] = 0;
        }
    }
    for( i = 0 ; i < N ; i++ ){
        work[i*(N+1)] = 1;
    }

    /* (R*A)*S': A <- work2 * work */

    memset( A, 0, M*N );
    dgemm_( &NoTranspose, &Transpose, &M, &N, &N, &c__1,
           work2, &N, work, &N, &c__1, A, &N ); 
    
    free( tau );
    
}

void ddiago( double* A, int M, int N, double cond ) {

    int i, end;

    end = MIN( M, N );
    A[0] = cond;
    for( i = 1 ; i < end ; i++ ) {
        A[ i*M + i] = 1;
    }
    
}
