#include <string.h>


/*
c     Compute 2D Haar transform of a matrix
c
c     Params:
c     A: input matrix M*N, double precision
c     B: output matrix M*N, double precision
c     W: work matrix M*N, double precision
c     M: nb of lines of the matrix, integer
c     N: nb of columns of the matrix, integer
c     LDA: leading dimension of A
c     LDB: leading dimension of B

c     TODO ASSUME M AND N ARE POWERS OF TWO
c     TRANS = 'T': column-major, 'N': row-major

c     Name:
c     d double
c     ha haar
c     mt matrix transform
c     2 2D
*/

void dhamt2_( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    int i, j, k;

    /* dim 1 */
    for( k = 0 ; k < M ; k++ ) {
        i = j = 0;
        while( i < N-1 ) { 
            W[ k*ldb + j] = ( A[k*lda + i] + A[ k*lda + i+1] ) / 2.0;
            i += 2;
            j++;
        }
        i = 0;
        while( i < N-1 ) {
            W[ k*ldb + j] = ( A[k*lda + i] - A[ k*lda + i+1] ) / 2.0;
            i += 2;
            j++;
        }
    }
    /* dim 2 */
    /* First term of the sum */
    k = i = 0;
    while( k < M-1 ) {
        memcpy( &(B[i*ldb]), &(W[k*ldb]), N*sizeof( double ) );
        k += 2;
        i++;
    }
    k = 0;
    while( k < M-1 ) {
        memcpy( &(B[i*ldb]), &(W[k*ldb]), N*sizeof( double ) );
        k += 2;
        i++;
    }
    
    /* Second term of the sum */
    for( k = 0, i = 0 ; k < M ; k+=2, i++ ) { /* upper half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] += W[(k+1)*ldb + j] ;
            B[i*ldb + j] /= 2;
        }
    }
    for( k = 0, i = M/2 ; k < M ; k+=2, i++ ) { /* lower half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] -= W[(k+1)*ldb + j] ;
            B[i*ldb + j] /= 2;
        }
    }
}
