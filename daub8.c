/* This file is part of pawt.
 *
 * Various implementations of the 2D Daubechies D8 transform dda4mt2.
 *
 * Copyright 2018 LIPN, CNRS UMR 7030, Université Paris 13, 
 *                Sorbonne-Paris-Cité. All rights reserved.
 * Author: see AUTHORS
 * Licence: GPL-3.0, see COPYING for details.
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <x86intrin.h>
#include <math.h>

#include "daub8.h"


/*
c     Compute 2D Daubechies D8 transform of a matrix
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
c     da8 Daubechies D8
c     mt matrix transform
c     2 2D
*/

#ifdef  __cplusplus
void dda8mt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda8mt2_initial( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h[8];
    double g[8];
    int i, j, k;

    dGetCoeffs8( h, g );

    /* dim 1 */

    for( k = 0 ; k < M ; k++ ) {
        i = j = 0;
        while( i < N-1 ) { 
            W[ k*N + j] = h[0] * A[k*lda + i] + h[1] * A[ k*lda + i+1] + h[2] * A[k*lda + ( i + 2 ) % N ] + h[3] * A[ k*lda + ( i + 3 ) % N ]
                + h[4] * A[k*lda + ( i + 4 ) % N ]  + h[5] * A[k*lda + ( i + 5 ) % N ] + h[6] * A[k*lda + ( i + 6 ) % N ] + h[7] * A[k*lda + ( i + 7 ) % N ];
            i += 2;
            j++;
        }
        i = 0;
        while( i < N-1 ) {
            W[ k*N + j] = g[0] * A[k*lda + i] + g[1] * A[ k*lda + i+1] + g[2] * A[k*lda + ( i + 2 ) % N ] + g[3] * A[ k*lda + ( i + 3 ) % N ]
                + g[4] * A[k*lda + ( i + 4 ) % N ]  + g[5] * A[k*lda + ( i + 5 ) % N ] + g[6] * A[k*lda + ( i + 6 ) % N ] + g[7] * A[k*lda + ( i + 7 ) % N ];
            i += 2;
            j++;
        }
    }
    
    /* dim 2 */

    for( k = 0, i = 0 ; k < M ; k+=2, i++ ) { /* upper half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] = h[0] * W[k*N + j] + h[1] * W[(k+1)*N + j] + h[2] * W[((k+2)%M)*N + j] + h[3] * W[((k+3)%M)*N + j]
                +  h[4] * W[((k+4)%M)*N + j] + h[5] * W[((k+5)%M)*N + j] + h[6] * W[((k+6)%M)*N + j] + h[7] * W[((k+7)%M)*N + j];
        }
    }
    for( k = 0, i = M/2; k < M ; k+=2, i++ ) { /* lower half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] = g[0] * W[k*N + j] + g[1] * W[(k+1)*N + j] + g[2] * W[((k+2)%M)*N + j] + g[3] * W[((k+3)%M)*N + j]
                +  g[4] * W[((k+4)%M)*N + j] + g[5] * W[((k+5)%M)*N + j] + g[6] * W[((k+6)%M)*N + j] + g[7] * W[((k+7)%M)*N + j];
        }
    }
}
 
#ifdef  __cplusplus
void dda8mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda8mt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h[8];
    double g[8];
    int i, j, k;

    dGetCoeffs8( h, g );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < (N / 2) ; i++ ){ 
            W[ j*N + i] =  h[0] * A[j*lda + 2*i] + h[1] * A[ j*lda + 2*i + 1] 
                + h[2] * A[j*lda + (2*i + 2)%N] + h[3] * A[ j*lda + (2*i + 3)%N]
                + h[4] * A[j*lda + (2*i + 4)%N] + h[5] * A[ j*lda + (2*i + 5)%N]
                + h[6] * A[j*lda + (2*i + 6)%N] + h[7] * A[ j*lda + (2*i + 7)%N]; // TODO virer les modulos ?
            W[ j*N + i + N/2] =  g[0] * A[j*lda + 2*i] + g[1] * A[ j*lda + 2*i + 1] 
                + g[2] * A[j*lda + (2*i + 2)%N] + g[3] * A[ j*lda + (2*i + 3)%N]
                + g[4] * A[j*lda + (2*i + 4)%N] + g[5] * A[ j*lda + (2*i + 5)%N]
                + g[6] * A[j*lda + (2*i + 6)%N] + g[7] * A[ j*lda + (2*i + 7)%N];
        }
   }
    
    /* dim 2 */
    
    for( j = 0 ; j < (M / 2) ; j++ ){ 
        for( i = 0 ; i < N ; i++ ){
            B[i + j * ldb] = h[0] * W[2*j*N +i] + h[1] * W[ (2*j+1)*N + i] 
                + h[2] * W[((2*j+2)%M)*N + i] + h[3] * W[ ((2*j+3)%M)*N + i]
                + h[4] * W[((2*j+4)%M)*N + i] + h[5] * W[ ((2*j+5)%M)*N + i]
                + h[6] * W[((2*j+6)%M)*N + i] + h[7] * W[ ((2*j+7)%M)*N + i];
            B[i + (j+M/2) * ldb ] = g[0] * W[2*j*N +i] + g[1] * W[ (2*j+1)*N + i] 
                + g[2] * W[((2*j+2)%M)*N + i] + g[3] * W[ ((2*j+3)%M)*N + i]
                + g[4] * W[((2*j+4)%M)*N + i] + g[5] * W[ ((2*j+5)%M)*N + i]
                + g[6] * W[((2*j+6)%M)*N + i] + g[7] * W[ ((2*j+7)%M)*N + i];
        }
    }
    
}
 
void dda8mt2_avx( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {

    double h[8];
    double g[8];
    int i, j, k;

    dGetCoeffs8( h, g );

    __m256d a[8], ah[8], ag[8], w[8], s[4];

    for( k = 0 ; k < 8 ; k++ ) {
        ah[k] = _mm256_set1_pd( h[k] );
        ag[k] = _mm256_set1_pd( g[k] );
    }

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < (N / 2) ; i+=4 ){ 

            for( k = 0 ; k < 8 ; k++ ) {
                a[k] = _mm256_set_pd( A[j*lda + ( 2*(i + 3) + k ) % N], A[j*lda + ( 2*(i + 2) + k ) % N], A[j*lda + ( 2*(i + 1) + k ) % N ], A[j*lda + ( 2*i + k ) % N] );
            }

            for( k = 0 ; k < 8 ; k++ ) {
                w[k] = _mm256_mul_pd( a[k], ah[k] );
            }
            for( k = 0 ; k < 4 ; k++ ) {
                s[k] = _mm256_add_pd( w[2*k], w[2*k + 1]);
            }
            w[0] = _mm256_add_pd( s[0], s[1] );
            w[1] = _mm256_add_pd( s[2], s[3] );
            s[0] = _mm256_add_pd( w[0], w[1] );

            _mm256_storeu_pd( &W[ j*ldb + i], s[0] );             

            for( k = 0 ; k < 8 ; k++ ) {
                w[k] = _mm256_mul_pd( a[k], ag[k] );
            }
            for( k = 0 ; k < 4 ; k++ ) {
                s[k] = _mm256_add_pd( w[2*k], w[2*k + 1]);
            }
            w[0] = _mm256_add_pd( s[0], s[1] );
            w[1] = _mm256_add_pd( s[2], s[3] );
            s[0] = _mm256_add_pd( w[0], w[1] );

            _mm256_storeu_pd( &W[ j*ldb + i + N / 2], s[0] );             

        }
    }

    /* dim 2 */
    
    for( j = 0 ; j < (M / 2) ; j++ ){ 
        for( i = 0 ; i < N ; i+=4 ){

            for( k = 0 ; k < 8 ; k++ ) {
                a[k] = _mm256_loadu_pd( &W[ ( ( 2*j + k) % M ) * N + i] );
            }

            for( k = 0 ; k < 8 ; k++ ) {
                w[k] = _mm256_mul_pd( a[k], ah[k] );
            }
            for( k = 0 ; k < 4 ; k++ ) {
                s[k] = _mm256_add_pd( w[2*k], w[2*k + 1]);
            }
            w[0] = _mm256_add_pd( s[0], s[1] );
            w[1] = _mm256_add_pd( s[2], s[3] );
            s[0] = _mm256_add_pd( w[0], w[1] );

            _mm256_storeu_pd( &B[ j*ldb + i], s[0] );             

            for( k = 0 ; k < 8 ; k++ ) {
                w[k] = _mm256_mul_pd( a[k], ag[k] );
            }
            for( k = 0 ; k < 4 ; k++ ) {
                s[k] = _mm256_add_pd( w[2*k], w[2*k + 1]);
            }
            w[0] = _mm256_add_pd( s[0], s[1] );
            w[1] = _mm256_add_pd( s[2], s[3] );
            s[0] = _mm256_add_pd( w[0], w[1] );

            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], s[0] );             

        }
    }
}
