/* This file is part of pawt.
 *
 * Various implementations of the 2D Daubechies D4 transform dda4mt2.
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

#ifdef __x86_64__
#include <x86intrin.h>
#endif // __x86_64__

#ifdef __aarch64__
#include <arm_neon.h>
#include "arm.h"
#endif // __aarch64__

#include <math.h>

#include "daub4.h"


/*
c     Compute 2D Daubechies D4 transform of a matrix
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
c     da4 Daubechies D4
c     mt matrix transform
c     2 2D
*/

#ifdef  __cplusplus
void dda4mt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda4mt2_initial( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j, k;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;

    /* dim 1 */

    for( k = 0 ; k < M ; k++ ) {
        i = j = 0;
        while( i < N-1 ) {
            W[ k*N + j] = h0 * A[k*lda + i] + h1 * A[ k*lda + i+1] + h2 * A[k*lda + ( i + 2 ) % N ] + h3 * A[ k*lda + ( i + 3 ) % N ];
            i += 2;
            j++;
        }
        i = 0;
        while( i < N-1 ) {
            W[ k*N + j] = g0 * A[k*lda + i] + g1 * A[ k*lda + i+1] + g2 * A[k*lda + ( i + 2 ) % N ] + g3 * A[ k*lda + ( i + 3 ) % N ];
            i += 2;
            j++;
        }
    }

    /* dim 2 */

    for( k = 0, i = 0 ; k < M ; k+=2, i++ ) { /* upper half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] = h0 * W[k*N + j] + h1 * W[(k+1)*N + j] + h2 * W[((k+2)%M)*N + j] + h3 * W[((k+3)%M)*N + j] ;
        }
    }
    for( k = 0, i = M/2; k < M ; k+=2, i++ ) { /* lower half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] = g0 * W[k*N + j] + g1 * W[(k+1)*N + j] + g2 * W[((k+2)%M)*N + j] + g3 * W[((k+3)%M)*N + j] ;
        }
    }
}

#ifdef  __cplusplus
void dda4mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda4mt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < (N / 2) ; i++ ){
            W[ j*N + i] =  h0 * A[j*lda + 2*i] + h1 * A[ j*lda + 2*i + 1]
                + h2 * A[j*lda + (2*i +2)%N] + h3 * A[ j*lda + (2*i + 3)%N];  // TODO virer les modulos ?
            W[ j*N + i + N/2] =  g0 * A[j*lda + 2*i] + g1 * A[ j*lda + 2*i + 1]
                + g2 * A[j*lda + (2*i + 2)%N] + g3 * A[ j*lda + (2*i + 3)%N];
       }
   }

    /* dim 2 */

    for( j = 0 ; j < (M / 2) ; j++ ){
        for( i = 0 ; i < N ; i++ ){
            B[i + j * ldb] = h0 * W[2*j*N +i] + h1 * W[ (2*j+1)*N + i]
                + h2 * W[((2*j+2)%M)*N + i] + h3 * W[ ((2*j+3)%M)*N + i];
            B[i + (j+M/2) * ldb ] = g0 * W[2*j*N +i] + g1 * W[ (2*j+1)*N + i]
                + g2 * W[((2*j+2)%M)*N + i] + g3 * W[ ((2*j+3)%M)*N + i];
      }
    }

}

#if defined( __SSE__ ) || defined( __aarch64__ )

void dda4mt2_sse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {

    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m128d a0, a1, a2, a3;
    __m128d w, w0, w1, w2, w3, s0, s1;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ){

            a0 = _mm_set_pd( A[j*lda + 2*(i + 1)], A[j*lda + 2*i] );
            a1 = _mm_set_pd( A[ j*lda + 2*(i + 1) + 1],   A[ j*lda + 2*i + 1] );
            a2 = _mm_set_pd( A[ j*lda + ( 2*(i + 1) + 2) %N],   A[ j*lda + ( 2*i + 2 )%N] );
            a3 = _mm_set_pd( A[ j*lda + ( 2*(i + 1) + 3 ) %N],   A[ j*lda + ( 2*i + 3 ) %N] );

            w0 = _mm_mul_pd( a0, ah0 );
            w1 = _mm_mul_pd( a1, ah1 );
            w2 = _mm_mul_pd( a2, ah2 );
            w3 = _mm_mul_pd( a3, ah3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &W[ j*ldb + i], w );

            w0 = _mm_mul_pd( a0, ag0 );
            w1 = _mm_mul_pd( a1, ag1 );
            w2 = _mm_mul_pd( a2, ag2 );
            w3 = _mm_mul_pd( a3, ag3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &W[ j*ldb + i + N/2], w );

        }
   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){
        for( i = 0 ; i < N ; i+=2 ){
            a0 = _mm_loadu_pd( &W[2*j*N + i] );
            a1 = _mm_loadu_pd( &W[(2*j+1)*N + i] );
            a2 = _mm_loadu_pd( &W[((2*j+2)%M)*N + i] );
            a3 = _mm_loadu_pd( &W[((2*j+3)%M)*N + i] );

            w0 = _mm_mul_pd( a0, ah0 );
            w1 = _mm_mul_pd( a1, ah1 );
            w2 = _mm_mul_pd( a2, ah2 );
            w3 = _mm_mul_pd( a3, ah3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ j*ldb + i], w );

            w0 = _mm_mul_pd( a0, ag0 );
            w1 = _mm_mul_pd( a1, ag1 );
            w2 = _mm_mul_pd( a2, ag2 );
            w3 = _mm_mul_pd( a3, ag3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ (j+M/2)*ldb + i], w );

        }
    }

}

void dda4mt2_sse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {

    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m128d a0, a1, a2, a3;
    __m128d w, w0, w1, w2, w3, s0, s1;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 - 1 ; i+=2 ){

            a0 = _mm_set_pd( A[j*lda + 2*(i + 1)], A[j*lda + 2*i] );
            a1 = _mm_set_pd( A[ j*lda + 2*(i + 1) + 1],   A[ j*lda + 2*i + 1 ] );
            a2 = _mm_set_pd( A[ j*lda + 2*(i + 1) + 2],   A[ j*lda + 2*i + 2 ] );
            a3 = _mm_set_pd( A[ j*lda + 2*(i + 1) + 3],   A[ j*lda + 2*i + 3 ] );

            w0 = _mm_mul_pd( a0, ah0 );
            w1 = _mm_mul_pd( a1, ah1 );
            w2 = _mm_mul_pd( a2, ah2 );
            w3 = _mm_mul_pd( a3, ah3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &W[ j*ldb + i], w );

            w0 = _mm_mul_pd( a0, ag0 );
            w1 = _mm_mul_pd( a1, ag1 );
            w2 = _mm_mul_pd( a2, ag2 );
            w3 = _mm_mul_pd( a3, ag3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &W[ j*ldb + i + N/2], w );
        }

        /* Peeling */

        a0 = _mm_set_pd( A[j*lda + N - 2 ], A[j*lda + N - 4] );
        a1 = _mm_set_pd( A[ j*lda + N - 1 ],   A[ j*lda + N - 3 ] );
        a2 = _mm_set_pd( A[ j*lda  ],   A[ j*lda + N - 2] );
        a3 = _mm_set_pd( A[ j*lda + 1],   A[ j*lda + N - 1 ] );

        w0 = _mm_mul_pd( a0, ah0 );
        w1 = _mm_mul_pd( a1, ah1 );
        w2 = _mm_mul_pd( a2, ah2 );
        w3 = _mm_mul_pd( a3, ah3 );

        s0 = _mm_add_pd( w0, w1);
        s1 = _mm_add_pd( w2, w3);
        w = _mm_add_pd( s0, s1 );

        _mm_storeu_pd( &W[ j*ldb + N/2 - 2], w );

        w0 = _mm_mul_pd( a0, ag0 );
        w1 = _mm_mul_pd( a1, ag1 );
        w2 = _mm_mul_pd( a2, ag2 );
        w3 = _mm_mul_pd( a3, ag3 );

        s0 = _mm_add_pd( w0, w1);
        s1 = _mm_add_pd( w2, w3);
        w = _mm_add_pd( s0, s1 );

        _mm_storeu_pd( &W[ j*ldb + N - 2], w );

   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 - 1 ; j++ ){
        for( i = 0 ; i < N ; i+=2 ){
            a0 = _mm_loadu_pd( &W[2*j*N + i] );
            a1 = _mm_loadu_pd( &W[(2*j+1)*N + i] );
            a2 = _mm_loadu_pd( &W[(2*j+2)*N + i] );
            a3 = _mm_loadu_pd( &W[(2*j+3)*N + i] );

            w0 = _mm_mul_pd( a0, ah0 );
            w1 = _mm_mul_pd( a1, ah1 );
            w2 = _mm_mul_pd( a2, ah2 );
            w3 = _mm_mul_pd( a3, ah3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ j*ldb + i], w );

            w0 = _mm_mul_pd( a0, ag0 );
            w1 = _mm_mul_pd( a1, ag1 );
            w2 = _mm_mul_pd( a2, ag2 );
            w3 = _mm_mul_pd( a3, ag3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ (j+M/2)*ldb + i], w );
        }
    }

    /* Peeling */

    for( i = 0 ; i < N ; i+=2 ){
        a0 = _mm_loadu_pd( &W[ ( M - 2 )*N + i] );
        a1 = _mm_loadu_pd( &W[ ( M - 1 )*N + i] );
        a2 = _mm_loadu_pd( &W[ ( 0 )*N + i] );
        a3 = _mm_loadu_pd( &W[ ( 1 )*N + i] );

        w0 = _mm_mul_pd( a0, ah0 );
        w1 = _mm_mul_pd( a1, ah1 );
        w2 = _mm_mul_pd( a2, ah2 );
        w3 = _mm_mul_pd( a3, ah3 );

        s0 = _mm_add_pd( w0, w1);
        s1 = _mm_add_pd( w2, w3);
        w = _mm_add_pd( s0, s1 );

        _mm_storeu_pd( &B[ (M/2 - 1)*ldb + i], w );

        w0 = _mm_mul_pd( a0, ag0 );
        w1 = _mm_mul_pd( a1, ag1 );
        w2 = _mm_mul_pd( a2, ag2 );
        w3 = _mm_mul_pd( a3, ag3 );

        s0 = _mm_add_pd( w0, w1);
        s1 = _mm_add_pd( w2, w3);
        w = _mm_add_pd( s0, s1 );

        _mm_storeu_pd( &B[ (M - 1)*ldb + i], w );
    }
}


void dda4mt2_sse_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {

    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m128d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m128d w, w0, w1, w2, w3, w0m, w1m, w2m, w3m, s0, s1, s2, s3;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );


    for( j = 0 ; j < M / 2; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ){

          a0 = _mm_set_pd( A[ (j*2)*lda + (2*i + 2)%N ], A[ (j*2)*lda + 2*i] );
          a1 = _mm_set_pd( A[ (j*2)*lda + (2*i + 3)%N ], A[ (j*2)*lda + 2*i + 1] );
          a2 = _mm_set_pd( A[ (j*2)*lda + (2*i + 4)%N ], A[ (j*2)*lda + (2*i + 2)%N ]);
          a3 = _mm_set_pd( A[ (j*2)*lda + (2*i + 5)%N ], A[ (j*2)*lda + (2*i + 3)%N ] );

          a4 = _mm_set_pd( A[ ((j*2+1)%M)*lda + (2*i + 2)%N ], A[ (j*2+1)*lda + 2*i] );
          a5 = _mm_set_pd( A[ ((j*2+1)%M)*lda + (2*i + 3)%N ], A[ (j*2+1)*lda + 2*i + 1] );
          a6 = _mm_set_pd( A[ ((j*2+1)%M)*lda + (2*i + 4)%N ], A[ (j*2+1)*lda + (2*i + 2)%N]);
          a7 = _mm_set_pd( A[ ((j*2+1)%M)*lda + (2*i + 5)%N ], A[ (j*2+1)*lda + (2*i + 3)%N ]);

          a8 = _mm_set_pd( A[ ((j*2+2)%M)*lda + (2*i + 2)%N ], A[ ((j*2+2)%M)*lda + 2*i] );
          a9 = _mm_set_pd( A[ ((j*2+2)%M)*lda + (2*i + 3)%N ], A[ ((j*2+2)%M)*lda + 2*i + 1] );
          aa = _mm_set_pd( A[ ((j*2+2)%M)*lda + (2*i + 4)%N ], A[ ((j*2+2)%M)*lda + (2*i + 2)%N] );
          ab = _mm_set_pd( A[ ((j*2+2)%M)*lda + (2*i + 5)%N ], A[ ((j*2+2)%M)*lda + (2*i + 3)%N] );

          ac = _mm_set_pd( A[((j*2+3)%M)*lda + (2*i + 2)%N ], A[((j*2+3)%M)*lda + 2*i] );
          ad = _mm_set_pd( A[((j*2+3)%M)*lda + (2*i + 3)%N ], A[((j*2+3)%M)*lda + 2*i + 1] );
          ae = _mm_set_pd( A[((j*2+3)%M)*lda + (2*i + 4)%N ], A[((j*2+3)%M)*lda + (2*i + 2)%N] );
          af = _mm_set_pd( A[((j*2+3)%M)*lda + (2*i + 5)%N ], A[((j*2+3)%M)*lda + (2*i + 3)%N] );

          /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

    	    s0 = _mm_mul_pd( a0, ah0 );
    	    s1 = _mm_mul_pd( a1, ah1 );
    	    s2 = _mm_mul_pd( a2, ah2 );
    	    s3 = _mm_mul_pd( a3, ah3 );

          w0 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a0, ag0 );
    	    s1 = _mm_mul_pd( a1, ag1 );
    	    s2 = _mm_mul_pd( a2, ag2 );
    	    s3 = _mm_mul_pd( a3, ag3 );

          w0m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a4, ah0 );
    	    s1 = _mm_mul_pd( a5, ah1 );
    	    s2 = _mm_mul_pd( a6, ah2 );
    	    s3 = _mm_mul_pd( a7, ah3 );

          w1 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a4, ag0 );
    	    s1 = _mm_mul_pd( a5, ag1 );
    	    s2 = _mm_mul_pd( a6, ag2 );
    	    s3 = _mm_mul_pd( a7, ag3 );

          w1m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a8, ah0 );
    	    s1 = _mm_mul_pd( a9, ah1 );
    	    s2 = _mm_mul_pd( aa, ah2 );
    	    s3 = _mm_mul_pd( ab, ah3 );

          w2 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a8, ag0 );
    	    s1 = _mm_mul_pd( a9, ag1 );
    	    s2 = _mm_mul_pd( aa, ag2 );
    	    s3 = _mm_mul_pd( ab, ag3 );

          w2m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( ac, ah0 );
    	    s1 = _mm_mul_pd( ad, ah1 );
    	    s2 = _mm_mul_pd( ae, ah2 );
    	    s3 = _mm_mul_pd( af, ah3 );

          w3 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( ac, ag0 );
    	    s1 = _mm_mul_pd( ad, ag1 );
    	    s2 = _mm_mul_pd( ae, ag2 );
    	    s3 = _mm_mul_pd( af, ag3 );

          w3m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

          /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

          s0 = _mm_add_pd( _mm_mul_pd( w2, ah2 ), _mm_mul_pd( w3, ah3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), _mm_mul_pd( w1, ah1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ j*ldb + i ], w );

          s0 = _mm_add_pd( _mm_mul_pd( w2, ag2 ), _mm_mul_pd( w3, ag3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0, ag0 ), _mm_mul_pd( w1, ag1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ (j + M/2)*ldb + i ], w );

          s0 = _mm_add_pd( _mm_mul_pd( w2m, ah2 ), _mm_mul_pd( w3m, ah3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0m, ah0 ), _mm_mul_pd( w1m, ah1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ j*ldb + i + N/2 ], w );

          s0 = _mm_add_pd( _mm_mul_pd( w2m, ag2 ), _mm_mul_pd( w3m, ag3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0m, ag0 ), _mm_mul_pd( w1m, ag1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ (j + M/2)*ldb + i + N / 2 ], w );
        }
   }
}

void dda4mt2_sse_reuse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {

    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m128d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m128d w, w0, w1, w2, w3, w0m, w1m, w2m, w3m, s0, s1, s2, s3;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );


    for( j = 0 ; j < M / 2 - 1; j++ ) {
        for( i = 0 ; i < N / 2 - 2 ; i+=2 ){

          a0 = _mm_set_pd( A[ (j*2)*lda + 2*i + 2 ], A[ (j*2)*lda + 2*i] );
          a1 = _mm_set_pd( A[ (j*2)*lda + 2*i + 3 ], A[ (j*2)*lda + 2*i + 1] );
          a2 = _mm_set_pd( A[ (j*2)*lda + 2*i + 4 ], A[ (j*2)*lda + 2*i + 2 ]);
          a3 = _mm_set_pd( A[ (j*2)*lda + 2*i + 5 ], A[ (j*2)*lda + 2*i + 3 ] );

          a4 = _mm_set_pd( A[ (j*2+1)*lda + 2*i + 2 ], A[ (j*2+1)*lda + 2*i] );
          a5 = _mm_set_pd( A[ (j*2+1)*lda + 2*i + 3 ], A[ (j*2+1)*lda + 2*i + 1] );
          a6 = _mm_set_pd( A[ (j*2+1)*lda + 2*i + 4 ], A[ (j*2+1)*lda + 2*i + 2 ]);
          a7 = _mm_set_pd( A[ (j*2+1)*lda + 2*i + 5 ], A[ (j*2+1)*lda + 2*i + 3 ]);

          a8 = _mm_set_pd( A[ (j*2+2)*lda + 2*i + 2 ], A[ (j*2+2)*lda + 2*i] );
          a9 = _mm_set_pd( A[ (j*2+2)*lda + 2*i + 3 ], A[ (j*2+2)*lda + 2*i + 1] );
          aa = _mm_set_pd( A[ (j*2+2)*lda + 2*i + 4 ], A[ (j*2+2)*lda + 2*i + 2] );
          ab = _mm_set_pd( A[ (j*2+2)*lda + 2*i + 5 ], A[ (j*2+2)*lda + 2*i + 3] );

          ac = _mm_set_pd( A[(j*2+3)*lda + 2*i + 2 ], A[(j*2+3)*lda + 2*i] );
          ad = _mm_set_pd( A[(j*2+3)*lda + 2*i + 3 ], A[(j*2+3)*lda + 2*i + 1] );
          ae = _mm_set_pd( A[(j*2+3)*lda + 2*i + 4 ], A[(j*2+3)*lda + 2*i + 2] );
          af = _mm_set_pd( A[(j*2+3)*lda + 2*i + 5 ], A[(j*2+3)*lda + 2*i + 3] );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

    	    s0 = _mm_mul_pd( a0, ah0 );
    	    s1 = _mm_mul_pd( a1, ah1 );
    	    s2 = _mm_mul_pd( a2, ah2 );
    	    s3 = _mm_mul_pd( a3, ah3 );

          w0 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a0, ag0 );
    	    s1 = _mm_mul_pd( a1, ag1 );
    	    s2 = _mm_mul_pd( a2, ag2 );
    	    s3 = _mm_mul_pd( a3, ag3 );

          w0m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a4, ah0 );
    	    s1 = _mm_mul_pd( a5, ah1 );
    	    s2 = _mm_mul_pd( a6, ah2 );
    	    s3 = _mm_mul_pd( a7, ah3 );

          w1 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a4, ag0 );
    	    s1 = _mm_mul_pd( a5, ag1 );
    	    s2 = _mm_mul_pd( a6, ag2 );
    	    s3 = _mm_mul_pd( a7, ag3 );

          w1m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a8, ah0 );
    	    s1 = _mm_mul_pd( a9, ah1 );
    	    s2 = _mm_mul_pd( aa, ah2 );
    	    s3 = _mm_mul_pd( ab, ah3 );

          w2 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( a8, ag0 );
    	    s1 = _mm_mul_pd( a9, ag1 );
    	    s2 = _mm_mul_pd( aa, ag2 );
    	    s3 = _mm_mul_pd( ab, ag3 );

          w2m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( ac, ah0 );
    	    s1 = _mm_mul_pd( ad, ah1 );
    	    s2 = _mm_mul_pd( ae, ah2 );
    	    s3 = _mm_mul_pd( af, ah3 );

          w3 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

    	    s0 = _mm_mul_pd( ac, ag0 );
    	    s1 = _mm_mul_pd( ad, ag1 );
    	    s2 = _mm_mul_pd( ae, ag2 );
    	    s3 = _mm_mul_pd( af, ag3 );

          w3m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

          /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

          s0 = _mm_add_pd( _mm_mul_pd( w2, ah2 ), _mm_mul_pd( w3, ah3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), _mm_mul_pd( w1, ah1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ j*ldb + i ], w );

          s0 = _mm_add_pd( _mm_mul_pd( w2, ag2 ), _mm_mul_pd( w3, ag3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0, ag0 ), _mm_mul_pd( w1, ag1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ (j + M/2)*ldb + i ], w );

          s0 = _mm_add_pd( _mm_mul_pd( w2m, ah2 ), _mm_mul_pd( w3m, ah3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0m, ah0 ), _mm_mul_pd( w1m, ah1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ j*ldb + i + N/2 ], w );

          s0 = _mm_add_pd( _mm_mul_pd( w2m, ag2 ), _mm_mul_pd( w3m, ag3 ) );
          s1 = _mm_add_pd( _mm_mul_pd( w0m, ag0 ), _mm_mul_pd( w1m, ag1 ) );
          w = _mm_add_pd( s0, s1 );
          _mm_storeu_pd( &B[ (j + M/2)*ldb + i + N / 2 ], w );
        }

        /* Peeling: lines */

        a0 = _mm_set_pd( A[ (j*2)*lda + N - 2 ], A[ (j*2)*lda + N - 4] );
        a1 = _mm_set_pd( A[ (j*2)*lda + N - 1 ], A[ (j*2)*lda + N - 3] );
        a2 = _mm_set_pd( A[ (j*2)*lda + 0 ], A[ (j*2)*lda + N - 2 ]);
        a3 = _mm_set_pd( A[ (j*2)*lda + 1 ], A[ (j*2)*lda + N - 1 ] );

        a4 = _mm_set_pd( A[ (j*2+1)*lda + N - 2 ], A[ (j*2+1)*lda + N - 4] );
        a5 = _mm_set_pd( A[ (j*2+1)*lda + N - 1 ], A[ (j*2+1)*lda + N - 3] );
        a6 = _mm_set_pd( A[ (j*2+1)*lda + 0 ], A[ (j*2+1)*lda + N - 2 ]);
        a7 = _mm_set_pd( A[ (j*2+1)*lda + 1 ], A[ (j*2+1)*lda + N - 1 ]);

        a8 = _mm_set_pd( A[ (j*2+2)*lda + N - 2 ], A[ (j*2+2)*lda + N - 4] );
        a9 = _mm_set_pd( A[ (j*2+2)*lda + N - 1 ], A[ (j*2+2)*lda + N - 3] );
        aa = _mm_set_pd( A[ (j*2+2)*lda + 0 ], A[ (j*2+2)*lda + N - 2] );
        ab = _mm_set_pd( A[ (j*2+2)*lda + 1 ], A[ (j*2+2)*lda + N - 1] );

        ac = _mm_set_pd( A[(j*2+3)*lda + N - 2 ], A[(j*2+3)*lda + N - 4] );
        ad = _mm_set_pd( A[(j*2+3)*lda + N - 1 ], A[(j*2+3)*lda + N - 3] );
        ae = _mm_set_pd( A[(j*2+3)*lda + 0 ], A[(j*2+3)*lda + N - 2] );
        af = _mm_set_pd( A[(j*2+3)*lda + 1 ], A[(j*2+3)*lda + N - 1] );

          /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

        s0 = _mm_mul_pd( a0, ah0 );
        s1 = _mm_mul_pd( a1, ah1 );
        s2 = _mm_mul_pd( a2, ah2 );
        s3 = _mm_mul_pd( a3, ah3 );

        w0 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( a0, ag0 );
        s1 = _mm_mul_pd( a1, ag1 );
        s2 = _mm_mul_pd( a2, ag2 );
        s3 = _mm_mul_pd( a3, ag3 );

        w0m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( a4, ah0 );
        s1 = _mm_mul_pd( a5, ah1 );
        s2 = _mm_mul_pd( a6, ah2 );
        s3 = _mm_mul_pd( a7, ah3 );

        w1 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( a4, ag0 );
        s1 = _mm_mul_pd( a5, ag1 );
        s2 = _mm_mul_pd( a6, ag2 );
        s3 = _mm_mul_pd( a7, ag3 );

        w1m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( a8, ah0 );
        s1 = _mm_mul_pd( a9, ah1 );
        s2 = _mm_mul_pd( aa, ah2 );
        s3 = _mm_mul_pd( ab, ah3 );

        w2 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( a8, ag0 );
        s1 = _mm_mul_pd( a9, ag1 );
        s2 = _mm_mul_pd( aa, ag2 );
        s3 = _mm_mul_pd( ab, ag3 );

        w2m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( ac, ah0 );
        s1 = _mm_mul_pd( ad, ah1 );
        s2 = _mm_mul_pd( ae, ah2 );
        s3 = _mm_mul_pd( af, ah3 );

        w3 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        s0 = _mm_mul_pd( ac, ag0 );
        s1 = _mm_mul_pd( ad, ag1 );
        s2 = _mm_mul_pd( ae, ag2 );
        s3 = _mm_mul_pd( af, ag3 );

        w3m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

        /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

        s0 = _mm_add_pd( _mm_mul_pd( w2, ah2 ), _mm_mul_pd( w3, ah3 ) );
        s1 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), _mm_mul_pd( w1, ah1 ) );
        w = _mm_add_pd( s0, s1 );
        _mm_storeu_pd( &B[ j*ldb + N/2 - 2 ], w );

        s0 = _mm_add_pd( _mm_mul_pd( w2, ag2 ), _mm_mul_pd( w3, ag3 ) );
        s1 = _mm_add_pd( _mm_mul_pd( w0, ag0 ), _mm_mul_pd( w1, ag1 ) );
        w = _mm_add_pd( s0, s1 );
        _mm_storeu_pd( &B[ (j + M/2)*ldb + N/2 - 2 ], w );

        s0 = _mm_add_pd( _mm_mul_pd( w2m, ah2 ), _mm_mul_pd( w3m, ah3 ) );
        s1 = _mm_add_pd( _mm_mul_pd( w0m, ah0 ), _mm_mul_pd( w1m, ah1 ) );
        w = _mm_add_pd( s0, s1 );
        _mm_storeu_pd( &B[ j*ldb + N - 2 ], w );

        s0 = _mm_add_pd( _mm_mul_pd( w2m, ag2 ), _mm_mul_pd( w3m, ag3 ) );
        s1 = _mm_add_pd( _mm_mul_pd( w0m, ag0 ), _mm_mul_pd( w1m, ag1 ) );
        w = _mm_add_pd( s0, s1 );
        _mm_storeu_pd( &B[ (j + M/2)*ldb + N - 2 ], w );
   }

   /* Peeling: columns */

   for( i = 0 ; i < N / 2 - 2 ; i+=2 ){

     a0 = _mm_set_pd( A[ (M - 2)*lda + 2*i + 2 ], A[ (M - 2)*lda + 2*i] );
     a1 = _mm_set_pd( A[ (M - 2)*lda + 2*i + 3 ], A[ (M - 2)*lda + 2*i + 1] );
     a2 = _mm_set_pd( A[ (M - 2)*lda + 2*i + 4 ], A[ (M - 2)*lda + 2*i + 2 ]);
     a3 = _mm_set_pd( A[ (M - 2)*lda + 2*i + 5 ], A[ (M - 2)*lda + 2*i + 3 ] );

     a4 = _mm_set_pd( A[ (M - 1)*lda + 2*i + 2 ], A[ (M - 1)*lda + 2*i] );
     a5 = _mm_set_pd( A[ (M - 1)*lda + 2*i + 3 ], A[ (M - 1)*lda + 2*i + 1] );
     a6 = _mm_set_pd( A[ (M - 1)*lda + 2*i + 4 ], A[ (M - 1)*lda + 2*i + 2 ]);
     a7 = _mm_set_pd( A[ (M - 1)*lda + 2*i + 5 ], A[ (M - 1)*lda + 2*i + 3 ]);

     a8 = _mm_set_pd( A[ (0)*lda + 2*i + 2 ], A[ (0)*lda + 2*i] );
     a9 = _mm_set_pd( A[ (0)*lda + 2*i + 3 ], A[ (0)*lda + 2*i + 1] );
     aa = _mm_set_pd( A[ (0)*lda + 2*i + 4 ], A[ (0)*lda + 2*i + 2] );
     ab = _mm_set_pd( A[ (0)*lda + 2*i + 5 ], A[ (0)*lda + 2*i + 3] );

     ac = _mm_set_pd( A[(1)*lda + 2*i + 2 ], A[(1)*lda + 2*i] );
     ad = _mm_set_pd( A[(1)*lda + 2*i + 3 ], A[(1)*lda + 2*i + 1] );
     ae = _mm_set_pd( A[(1)*lda + 2*i + 4 ], A[(1)*lda + 2*i + 2] );
     af = _mm_set_pd( A[(1)*lda + 2*i + 5 ], A[(1)*lda + 2*i + 3] );

       /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

      s0 = _mm_mul_pd( a0, ah0 );
      s1 = _mm_mul_pd( a1, ah1 );
      s2 = _mm_mul_pd( a2, ah2 );
      s3 = _mm_mul_pd( a3, ah3 );

       w0 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( a0, ag0 );
      s1 = _mm_mul_pd( a1, ag1 );
      s2 = _mm_mul_pd( a2, ag2 );
      s3 = _mm_mul_pd( a3, ag3 );

       w0m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( a4, ah0 );
      s1 = _mm_mul_pd( a5, ah1 );
      s2 = _mm_mul_pd( a6, ah2 );
      s3 = _mm_mul_pd( a7, ah3 );

       w1 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( a4, ag0 );
      s1 = _mm_mul_pd( a5, ag1 );
      s2 = _mm_mul_pd( a6, ag2 );
      s3 = _mm_mul_pd( a7, ag3 );

       w1m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( a8, ah0 );
      s1 = _mm_mul_pd( a9, ah1 );
      s2 = _mm_mul_pd( aa, ah2 );
      s3 = _mm_mul_pd( ab, ah3 );

       w2 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( a8, ag0 );
      s1 = _mm_mul_pd( a9, ag1 );
      s2 = _mm_mul_pd( aa, ag2 );
      s3 = _mm_mul_pd( ab, ag3 );

       w2m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( ac, ah0 );
      s1 = _mm_mul_pd( ad, ah1 );
      s2 = _mm_mul_pd( ae, ah2 );
      s3 = _mm_mul_pd( af, ah3 );

       w3 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

      s0 = _mm_mul_pd( ac, ag0 );
      s1 = _mm_mul_pd( ad, ag1 );
      s2 = _mm_mul_pd( ae, ag2 );
      s3 = _mm_mul_pd( af, ag3 );

     w3m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

     /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

     s0 = _mm_add_pd( _mm_mul_pd( w2, ah2 ), _mm_mul_pd( w3, ah3 ) );
     s1 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), _mm_mul_pd( w1, ah1 ) );
     w = _mm_add_pd( s0, s1 );
     _mm_storeu_pd( &B[ (M/2 - 1)*ldb + i ], w );

     s0 = _mm_add_pd( _mm_mul_pd( w2, ag2 ), _mm_mul_pd( w3, ag3 ) );
     s1 = _mm_add_pd( _mm_mul_pd( w0, ag0 ), _mm_mul_pd( w1, ag1 ) );
     w = _mm_add_pd( s0, s1 );
     _mm_storeu_pd( &B[ (M - 1)*ldb + i ], w );

     s0 = _mm_add_pd( _mm_mul_pd( w2m, ah2 ), _mm_mul_pd( w3m, ah3 ) );
     s1 = _mm_add_pd( _mm_mul_pd( w0m, ah0 ), _mm_mul_pd( w1m, ah1 ) );
     w = _mm_add_pd( s0, s1 );
     _mm_storeu_pd( &B[ (M/2 - 1)*ldb + i + N/2 ], w );

     s0 = _mm_add_pd( _mm_mul_pd( w2m, ag2 ), _mm_mul_pd( w3m, ag3 ) );
     s1 = _mm_add_pd( _mm_mul_pd( w0m, ag0 ), _mm_mul_pd( w1m, ag1 ) );
     w = _mm_add_pd( s0, s1 );
     _mm_storeu_pd( &B[ (M - 1)*ldb + i + N / 2 ], w );
   }

   /* Peeling: last corner */

   a0 = _mm_set_pd( A[ (M - 2)*lda + N - 2 ], A[ (M - 2)*lda + N - 4] );
   a1 = _mm_set_pd( A[ (M - 2)*lda + N - 1 ], A[ (M - 2)*lda + N - 3] );
   a2 = _mm_set_pd( A[ (M - 2)*lda + 0 ], A[ (M - 2)*lda + N - 2 ]);
   a3 = _mm_set_pd( A[ (M - 2)*lda + 1 ], A[ (M - 2)*lda + N - 1 ] );

   a4 = _mm_set_pd( A[ (M - 1)*lda + N - 2 ], A[ (M - 1)*lda + N - 4] );
   a5 = _mm_set_pd( A[ (M - 1)*lda + N - 1 ], A[ (M - 1)*lda + N - 3] );
   a6 = _mm_set_pd( A[ (M - 1)*lda + 0 ], A[ (M - 1)*lda + N - 2 ]);
   a7 = _mm_set_pd( A[ (M - 1)*lda + 1 ], A[ (M - 1)*lda + N - 1 ]);

   a8 = _mm_set_pd( A[ (0)*lda + N - 2 ], A[ (0)*lda + N - 4] );
   a9 = _mm_set_pd( A[ (0)*lda + N - 1 ], A[ (0)*lda + N - 3] );
   aa = _mm_set_pd( A[ (0)*lda + 0 ], A[ (0)*lda + N - 2] );
   ab = _mm_set_pd( A[ (0)*lda + 1 ], A[ (0)*lda + N - 1] );

   ac = _mm_set_pd( A[(1)*lda + N - 2 ], A[(1)*lda + N - 4] );
   ad = _mm_set_pd( A[(1)*lda + N - 1 ], A[(1)*lda + N - 3] );
   ae = _mm_set_pd( A[(1)*lda + 0 ], A[(1)*lda + N - 2] );
   af = _mm_set_pd( A[(1)*lda + 1 ], A[(1)*lda + N - 1] );

   /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

  s0 = _mm_mul_pd( a0, ah0 );
  s1 = _mm_mul_pd( a1, ah1 );
  s2 = _mm_mul_pd( a2, ah2 );
  s3 = _mm_mul_pd( a3, ah3 );

   w0 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( a0, ag0 );
  s1 = _mm_mul_pd( a1, ag1 );
  s2 = _mm_mul_pd( a2, ag2 );
  s3 = _mm_mul_pd( a3, ag3 );

   w0m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( a4, ah0 );
  s1 = _mm_mul_pd( a5, ah1 );
  s2 = _mm_mul_pd( a6, ah2 );
  s3 = _mm_mul_pd( a7, ah3 );

   w1 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( a4, ag0 );
  s1 = _mm_mul_pd( a5, ag1 );
  s2 = _mm_mul_pd( a6, ag2 );
  s3 = _mm_mul_pd( a7, ag3 );

   w1m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( a8, ah0 );
  s1 = _mm_mul_pd( a9, ah1 );
  s2 = _mm_mul_pd( aa, ah2 );
  s3 = _mm_mul_pd( ab, ah3 );

   w2 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( a8, ag0 );
  s1 = _mm_mul_pd( a9, ag1 );
  s2 = _mm_mul_pd( aa, ag2 );
  s3 = _mm_mul_pd( ab, ag3 );

   w2m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( ac, ah0 );
  s1 = _mm_mul_pd( ad, ah1 );
  s2 = _mm_mul_pd( ae, ah2 );
  s3 = _mm_mul_pd( af, ah3 );

   w3 = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

  s0 = _mm_mul_pd( ac, ag0 );
  s1 = _mm_mul_pd( ad, ag1 );
  s2 = _mm_mul_pd( ae, ag2 );
  s3 = _mm_mul_pd( af, ag3 );

 w3m = _mm_add_pd( _mm_add_pd( s0, s1 ), _mm_add_pd( s2, s3 ) );

 /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

 s0 = _mm_add_pd( _mm_mul_pd( w2, ah2 ), _mm_mul_pd( w3, ah3 ) );
 s1 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), _mm_mul_pd( w1, ah1 ) );
 w = _mm_add_pd( s0, s1 );
 _mm_storeu_pd( &B[ (M/2 - 1)*ldb + N / 2 - 2 ], w );

 s0 = _mm_add_pd( _mm_mul_pd( w2, ag2 ), _mm_mul_pd( w3, ag3 ) );
 s1 = _mm_add_pd( _mm_mul_pd( w0, ag0 ), _mm_mul_pd( w1, ag1 ) );
 w = _mm_add_pd( s0, s1 );
 _mm_storeu_pd( &B[ (M - 1)*ldb +  N / 2 - 2 ], w );

 s0 = _mm_add_pd( _mm_mul_pd( w2m, ah2 ), _mm_mul_pd( w3m, ah3 ) );
 s1 = _mm_add_pd( _mm_mul_pd( w0m, ah0 ), _mm_mul_pd( w1m, ah1 ) );
 w = _mm_add_pd( s0, s1 );
 _mm_storeu_pd( &B[ (M/2 - 1)*ldb + N - 2 ], w );

 s0 = _mm_add_pd( _mm_mul_pd( w2m, ag2 ), _mm_mul_pd( w3m, ag3 ) );
 s1 = _mm_add_pd( _mm_mul_pd( w0m, ag0 ), _mm_mul_pd( w1m, ag1 ) );
 w = _mm_add_pd( s0, s1 );
 _mm_storeu_pd( &B[ (M - 1)*ldb + N - 2 ], w );

}

#endif // __SSE__ || __aarch64__


#ifdef __AVX__
void dda4mt2_avx( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3;
    __m256d w, w0, w1, w2, w3, s0, s1;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){

            a0 = _mm256_set_pd( A[j*lda + ( 2*(i + 3) ) % N], A[j*lda + ( 2*(i + 2) ) % N], A[j*lda + 2*(i + 1)], A[j*lda + 2*i] );
            a1 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 1 ) % N],  A[ j*lda + ( 2*(i + 2) + 1 ) % N],  A[ j*lda + 2*(i + 1) + 1],   A[ j*lda + 2*i + 1] );
            a2 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 2 ) % N],  A[ j*lda + ( 2*(i + 2) + 2) %N],  A[ j*lda + ( 2*(i + 1) + 2) %N],   A[ j*lda + ( 2*i + 2 )%N] );
            a3 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 3 ) % N],  A[ j*lda + ( 2*(i + 2) + 3 ) % N],  A[ j*lda + ( 2*(i + 1) + 3 ) %N],   A[ j*lda + ( 2*i + 3 ) %N] );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ah1 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ah3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*ldb + i], w );

            w0 = _mm256_mul_pd( a0, ag0 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ag2 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*ldb + i + N/2], w );

        }
   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){
        for( i = 0 ; i < N ; i+=4 ){
            a0 = _mm256_loadu_pd( &W[2*j*N + i] );
            a1 = _mm256_loadu_pd( &W[(2*j+1)*N + i] );
            a2 = _mm256_loadu_pd( &W[((2*j+2)%M)*N + i] );
            a3 = _mm256_loadu_pd( &W[((2*j+3)%M)*N + i] );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ah1 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ah3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ j*ldb + i], w );

            w0 = _mm256_mul_pd( a0, ag0 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ag2 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], w );

        }
    }
}

void dda4mt2_avx_peel( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
   double h0, h1, h2, h3;
   double g0, g1, g2, g3;
   int i, j;

   __m256d a0, a1, a2, a3;
   __m256d w0, w1, w2, w3, s0, s1;
   __m256d ah0, ah1, ah2, ah3;
   __m256d ag0, ag1, ag2, ag3;

   dGetCoeffs4( &h0, &h1, &h2, &h3 );
   g0 = h3;
   g1 = -h2;
   g2 = h1;
   g3 = -h0;
   ah0 = _mm256_set1_pd( h0 );
   ah1 = _mm256_set1_pd( h1 );
   ah2 = _mm256_set1_pd( h2 );
   ah3 = _mm256_set1_pd( h3 );
   ag0 = _mm256_set1_pd( g0 );
   ag1 = _mm256_set1_pd( g1 );
   ag2 = _mm256_set1_pd( g2 );
   ag3 = _mm256_set1_pd( g3 );

   /* Dim 1 */

   for( j = 0 ; j < M ; j++ ) {
 	   for( i = 0 ; i < N / 2 -  2  ; i+= 4 ){
         a0  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  0  )], A[ j*lda + ( 2*(i +  2 ) +  0  )], A[ j*lda + ( 2*(i +  1 ) +  0  )], A[ j*lda + ( 2*(i +  0 ) +  0  )] );
         a1  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  1  )], A[ j*lda + ( 2*(i +  2 ) +  1  )], A[ j*lda + ( 2*(i +  1 ) +  1  )], A[ j*lda + ( 2*(i +  0 ) +  1  )] );
         a2  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  2  )], A[ j*lda + ( 2*(i +  2 ) +  2  )], A[ j*lda + ( 2*(i +  1 ) +  2  )], A[ j*lda + ( 2*(i +  0 ) +  2  )] );
         a3  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  3  )], A[ j*lda + ( 2*(i +  2 ) +  3  )], A[ j*lda + ( 2*(i +  1 ) +  3  )], A[ j*lda + ( 2*(i +  0 ) +  3  )] );

         w0 = _mm256_mul_pd( a0, ah0 );
         w1 = _mm256_mul_pd( a1, ah1 );
         w2 = _mm256_mul_pd( a2, ah2 );
         w3 = _mm256_mul_pd( a3, ah3 );

         s0 = _mm256_add_pd( w0, w1 );
         s1 = _mm256_add_pd( w2, w3 );
         s0 = _mm256_add_pd( s0, s1 );

         _mm256_storeu_pd( &W[ j*ldb + i], s0 );

         w0 = _mm256_mul_pd( a0, ag0 );
         w1 = _mm256_mul_pd( a1, ag1 );
         w2 = _mm256_mul_pd( a2, ag2 );
         w3 = _mm256_mul_pd( a3, ag3 );

         s0 = _mm256_add_pd( w0, w1 );
         s1 = _mm256_add_pd( w2, w3 );
         s0 = _mm256_add_pd( s0, s1 );

         _mm256_storeu_pd( &W[ j*ldb + i + N/2], s0 );
   	   }
   }

   /* Peeling */

   for( j = 0 ; j < M ; j++ ) {
       a0  = _mm256_set_pd( A[ j*lda + N - 2 ] , A[ j*lda + N - 4 ] , A[ j*lda + N - 6 ] , A[ j*lda + N - 8 ]  );
       a1  = _mm256_set_pd( A[ j*lda + N - 1 ] , A[ j*lda + N - 3 ] , A[ j*lda + N - 5 ] , A[ j*lda + N - 7 ]  );
       a2  = _mm256_set_pd( A[ j*lda + 0 ], A[ j*lda + N - 2 ] , A[ j*lda + N - 4 ] , A[ j*lda + N - 6 ]  );
       a3  = _mm256_set_pd( A[ j*lda + 1 ], A[ j*lda + N - 1 ] , A[ j*lda + N - 3 ] , A[ j*lda + N - 5 ]  );

       w0 = _mm256_mul_pd( a0, ah0 );
       w1 = _mm256_mul_pd( a1, ah1 );
       w2 = _mm256_mul_pd( a2, ah2 );
       w3 = _mm256_mul_pd( a3, ah3 );

       s0 = _mm256_add_pd( w0, w1 );
       s1 = _mm256_add_pd( w2, w3 );
       s0 = _mm256_add_pd( s0, s1 );

       _mm256_storeu_pd( &W[ j*ldb + N/2 - 4], s0 );

       w0 = _mm256_mul_pd( a0, ag0 );
       w1 = _mm256_mul_pd( a1, ag1 );
       w2 = _mm256_mul_pd( a2, ag2 );
       w3 = _mm256_mul_pd( a3, ag3 );

       s0 = _mm256_add_pd( w0, w1 );
       s1 = _mm256_add_pd( w2, w3 );
       s0 = _mm256_add_pd( s0, s1 );

       _mm256_storeu_pd( &W[ j*ldb + N - 4], s0 );
   }

   /* Dim 2 */

   for( j = 0 ; j < M / 2 - 1 ; j++ ) {
 	   for( i = 0 ; i < N  ; i+= 4 ){
       a0  = _mm256_loadu_pd( &W[( 2*j +0 ) * N + i] );
       a1  = _mm256_loadu_pd( &W[( 2*j +1 ) * N + i] );
       a2  = _mm256_loadu_pd( &W[( 2*j +2 ) * N + i] );
       a3  = _mm256_loadu_pd( &W[( 2*j +3 ) * N + i] );

       w0 = _mm256_mul_pd( a0, ah0 );
       w1 = _mm256_mul_pd( a1, ah1 );
       w2 = _mm256_mul_pd( a2, ah2 );
       w3 = _mm256_mul_pd( a3, ah3 );

       s0 = _mm256_add_pd( w0, w1 );
       s1 = _mm256_add_pd( w2, w3 );
       s0 = _mm256_add_pd( s0, s1 );

       _mm256_storeu_pd( &B[ j*ldb + i], s0 );

       w0 = _mm256_mul_pd( a0, ag0 );
       w1 = _mm256_mul_pd( a1, ag1 );
       w2 = _mm256_mul_pd( a2, ag2 );
       w3 = _mm256_mul_pd( a3, ag3 );

       s0 = _mm256_add_pd( w0, w1 );
       s1 = _mm256_add_pd( w2, w3 );
       s0 = _mm256_add_pd( s0, s1 );

       _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], s0 );
   	}
   }

   /* Peeling */

  for( i = 0 ; i < N ; i+= 4 ){
      a0  = _mm256_loadu_pd( &W[ ( M -  2 )* N + i] );
      a1  = _mm256_loadu_pd( &W[ ( M -  1 )* N + i] );
      a2  = _mm256_loadu_pd( &W[ (0 )* N + i] );
      a3  = _mm256_loadu_pd( &W[ (1 )* N + i] );

       w0 = _mm256_mul_pd( a0, ah0 );
       w1 = _mm256_mul_pd( a1, ah1 );
       w2 = _mm256_mul_pd( a2, ah2 );
       w3 = _mm256_mul_pd( a3, ah3 );

       s0 = _mm256_add_pd( w0, w1 );
       s1 = _mm256_add_pd( w2, w3 );
       s0 = _mm256_add_pd( s0, s1 );

       _mm256_storeu_pd( &B[ (M/2 - 1)*ldb + i], s0 );

       w0 = _mm256_mul_pd( a0, ag0 );
       w1 = _mm256_mul_pd( a1, ag1 );
       w2 = _mm256_mul_pd( a2, ag2 );
       w3 = _mm256_mul_pd( a3, ag3 );

       s0 = _mm256_add_pd( w0, w1 );
       s1 = _mm256_add_pd( w2, w3 );
       s0 = _mm256_add_pd( s0, s1 );

       _mm256_storeu_pd( &B[ (M - 1)*ldb + i], s0 );
   }
}


void dda4mt2_avx_gather( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3;
    __m256d w, w0, w1, w2, w3, s0, s1;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256i stride = _mm256_set_epi64x( 3*sizeof( double ), 2*sizeof( double ),
                                          sizeof( double ), 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 - 4 ; i+=4 ){
            a0 = _mm256_i64gather_pd( &A[j*lda + 2*i], stride, 2 );
            a1 = _mm256_i64gather_pd( &A[j*lda + 2*i + 1], stride, 2 );
            a2 = _mm256_i64gather_pd( &A[j*lda + 2*i + 2], stride, 2 );
            a3 = _mm256_i64gather_pd( &A[j*lda + 2*i + 3], stride, 2 );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ah1 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ah3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*N + i], w );

            w0 = _mm256_mul_pd( a0, ag0 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ag2 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*N + i + N/2], w );
        }

        /* The last column cannot be done with gather because of the folding */
        a0 = _mm256_i64gather_pd( &A[j*lda + 2*i], stride, 2 );
        a1 = _mm256_i64gather_pd( &A[j*lda + 2*i + 1], stride, 2 );
        a2 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 2 ) % N],  A[ j*lda + ( 2*(i + 2) + 2) %N],  A[ j*lda + ( 2*(i + 1) + 2) %N],   A[ j*lda + ( 2*i + 2 )%N] );
        a3 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 3 ) % N],  A[ j*lda + ( 2*(i + 2) + 3 ) % N],  A[ j*lda + ( 2*(i + 1) + 3 ) %N],   A[ j*lda + ( 2*i + 3 ) %N] );

        w0 = _mm256_mul_pd( a0, ah0 );
        w1 = _mm256_mul_pd( a1, ah1 );
        w2 = _mm256_mul_pd( a2, ah2 );
        w3 = _mm256_mul_pd( a3, ah3 );

        s0 = _mm256_add_pd( w0, w1);
        s1 = _mm256_add_pd( w2, w3);
        w = _mm256_add_pd( s0, s1 );

        _mm256_storeu_pd( &W[ j*N + i], w );

        w0 = _mm256_mul_pd( a0, ag0 );
        w1 = _mm256_mul_pd( a1, ag1 );
        w2 = _mm256_mul_pd( a2, ag2 );
        w3 = _mm256_mul_pd( a3, ag3 );

        s0 = _mm256_add_pd( w0, w1);
        s1 = _mm256_add_pd( w2, w3);
        w = _mm256_add_pd( s0, s1 );

        _mm256_storeu_pd( &W[ j*N + i + N/2], w );
   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){
        for( i = 0 ; i < N ; i+=4 ){
            a0 = _mm256_loadu_pd( &W[2*j*N + i] );
            a1 = _mm256_loadu_pd( &W[(2*j+1)*N + i] );
            a2 = _mm256_loadu_pd( &W[((2*j+2)%M)*N + i] );
            a3 = _mm256_loadu_pd( &W[((2*j+3)%M)*N + i] );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ah1 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ah3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ j*ldb + i], w );

            w0 = _mm256_mul_pd( a0, ag0 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ag2 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], w );

        }

    }

}
#endif // __AVX__

#ifdef __AVX2__
void dda4mt2_fma( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3;
    __m256d w, s0, s1;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){

            a0 = _mm256_set_pd( A[j*lda + ( 2*(i + 3) ) % N], A[j*lda + ( 2*(i + 2) ) % N], A[j*lda + 2*(i + 1)], A[j*lda + 2*i] );
            a1 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 1 ) % N],  A[ j*lda + ( 2*(i + 2) + 1 ) % N],  A[ j*lda + 2*(i + 1) + 1],   A[ j*lda + 2*i + 1] );
            a2 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 2 ) % N],  A[ j*lda + ( 2*(i + 2) + 2) %N],  A[ j*lda + ( 2*(i + 1) + 2) %N],   A[ j*lda + ( 2*i + 2 )%N] );
            a3 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 3 ) % N],  A[ j*lda + ( 2*(i + 2) + 3 ) % N],  A[ j*lda + ( 2*(i + 1) + 3 ) %N],   A[ j*lda + ( 2*i + 3 ) %N] );

            /* w = ( a0 * h0 + ( a1 * h1 ) ) + ( a2 * ( h2 + (a3 * h3))) */

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );

            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );

            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*ldb + i + N/2], w );

        }
   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){
        for( i = 0 ; i < N ; i+=4 ){
            a0 = _mm256_loadu_pd( &W[2*j*N + i] );
            a1 = _mm256_loadu_pd( &W[(2*j+1)*N + i] );
            a2 = _mm256_loadu_pd( &W[((2*j+2)%M)*N + i] );
            a3 = _mm256_loadu_pd( &W[((2*j+3)%M)*N + i] );

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );

            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );

            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], w );

        }
    }
}

/* Different order of operations */

void dda4mt2_fma2( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3;
    __m256d w, s0, s1;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){

            a0 = _mm256_set_pd( A[j*lda + ( 2*(i + 3) ) % N], A[j*lda + ( 2*(i + 2) ) % N], A[j*lda + 2*(i + 1)], A[j*lda + 2*i] );
            a1 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 1 ) % N],  A[ j*lda + ( 2*(i + 2) + 1 ) % N],  A[ j*lda + 2*(i + 1) + 1],   A[ j*lda + 2*i + 1] );
            a2 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 2 ) % N],  A[ j*lda + ( 2*(i + 2) + 2) %N],  A[ j*lda + ( 2*(i + 1) + 2) %N],   A[ j*lda + ( 2*i + 2 )%N] );
            a3 = _mm256_set_pd(  A[ j*lda + ( 2*(i + 3) + 3 ) % N],  A[ j*lda + ( 2*(i + 2) + 3 ) % N],  A[ j*lda + ( 2*(i + 1) + 3 ) %N],   A[ j*lda + ( 2*i + 3 ) %N] );

            /* Variant:
               w = a0 * h0 + ( a1 * h1 + ( a2 * h2 + (a3 * h3))) */

            s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
            s1 = _mm256_fmadd_pd( a1, ah1, s0 );
            w = _mm256_fmadd_pd( a0, ah0, s1 );

            _mm256_storeu_pd( &W[ j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
            s1 = _mm256_fmadd_pd( a1, ag1, s0 );
            w = _mm256_fmadd_pd( a0, ag0, s1 );

            _mm256_storeu_pd( &W[ j*ldb + i + N/2], w );

        }
   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){
        for( i = 0 ; i < N ; i+=4 ){
            a0 = _mm256_loadu_pd( &W[2*j*N + i] );
            a1 = _mm256_loadu_pd( &W[(2*j+1)*N + i] );
            a2 = _mm256_loadu_pd( &W[((2*j+2)%M)*N + i] );
            a3 = _mm256_loadu_pd( &W[((2*j+3)%M)*N + i] );

            s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
            s1 = _mm256_fmadd_pd( a1, ah1, s0 );
            w = _mm256_fmadd_pd( a0, ah0, s1 );

            _mm256_storeu_pd( &B[ j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
            s1 = _mm256_fmadd_pd( a1, ag1, s0 );
            w = _mm256_fmadd_pd( a0, ag0, s1 );

           _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], w );

        }
    }
}

/* Based on the order of operations performend by fma */

void dda4mt2_fma_reuse( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m256d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M / 2; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){

            a0 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + ( 2*i + 4 ) % N], A[(j*2)*lda + 2*i + 2 ], A[(j*2)*lda + 2*i] );
            a1 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + ( 2*i + 5 ) % N],  A[ (j*2)*lda + 2*i + 3 ],   A[ (j*2)*lda + 2*i + 1] );
            a2 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 8 ) % N],  A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + 2*i + 4 ],   A[ (j*2)*lda + 2*i + 2 ]);
            a3 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 9 ) % N],  A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + 2*i + 5 ],   A[ (j*2)*lda + 2*i + 3 ] );

            a4 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + ( 2*i + 4 ) % N], A[(j*2+1)*lda + 2*i + 2 ], A[(j*2+1)*lda + 2*i] );
            a5 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + ( 2*i + 5 ) % N],  A[ (j*2+1)*lda + 2*i + 3 ],   A[ (j*2+1)*lda + 2*i + 1] );
            a6 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 8 ) % N],  A[ (j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + 2*i + 4 ],   A[ (j*2+1)*lda + 2*i + 2 ]);
            a7 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 9 ) % N],  A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + 2*i + 5 ],   A[ (j*2+1)*lda + 2*i + 3 ] );

            a8 = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 4 ) % N], A[((j*2+2)%M)*lda + 2*i + 2 ], A[((j*2+2)%M)*lda + 2*i] );
            a9 = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 5 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 3 ],   A[ ((j*2+2)%M)*lda + 2*i + 1] );
            aa = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 4 ],   A[ ((j*2+2)%M)*lda + 2*i + 2 ]);
            ab = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 5 ],   A[ ((j*2+2)%M)*lda + 2*i + 3 ] );

            ac = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 4 ) % N], A[((j*2+3)%M)*lda + 2*i + 2 ], A[((j*2+3)%M)*lda + 2*i] );
            ad = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 5 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 3 ],   A[ ((j*2+3)%M)*lda + 2*i + 1] );
            ae = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 4 ],   A[ ((j*2+3)%M)*lda + 2*i + 2 ]);
            af = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 5 ],   A[ ((j*2+3)%M)*lda + 2*i + 3 ] );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );
            w0 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );
            w0m = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a4, ah0,  _mm256_mul_pd( a5, ah1 ) );
            s1 = _mm256_fmadd_pd( a6, ah2,  _mm256_mul_pd( a7, ah3 ) );
            w1 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a4, ag0,  _mm256_mul_pd( a5, ag1 ) );
            s1 = _mm256_fmadd_pd( a6, ag2,  _mm256_mul_pd( a7, ag3 ) );
            w1m = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a8, ah0,  _mm256_mul_pd( a9, ah1 ) );
            s1 = _mm256_fmadd_pd( aa, ah2,  _mm256_mul_pd( ab, ah3 ) );
            w2 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a8, ag0,  _mm256_mul_pd( a9, ag1 ) );
            s1 = _mm256_fmadd_pd( aa, ag2,  _mm256_mul_pd( ab, ag3 ) );
            w2m = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( ac, ah0,  _mm256_mul_pd( ad, ah1 ) );
            s1 = _mm256_fmadd_pd( ae, ah2,  _mm256_mul_pd( af, ah3 ) );
            w3 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( ac, ag0,  _mm256_mul_pd( ad, ag1 ) );
            s1 = _mm256_fmadd_pd( ae, ag2,  _mm256_mul_pd( af, ag3 ) );
            w3m = _mm256_add_pd( s0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
            s1 = _mm256_fmadd_pd( w1, ah1, s0 );
            w = _mm256_fmadd_pd( w0, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
            s1 = _mm256_fmadd_pd( w1, ag1, s0 );
            w = _mm256_fmadd_pd( w0, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
            s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
            w = _mm256_fmadd_pd( w0m, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

            s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
            s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
            w = _mm256_fmadd_pd( w0m, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );

        }
   }

}

void dda4mt2_fma_reuse_peel( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m256d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M / 2 - 1; j++ ) {
        for( i = 0 ; i < N / 2 - 2 ; i+=4 ){

            a0  = _mm256_set_pd( A[ (j*2 + 0 )*lda + 2*i +  6 ], A[ (j*2 + 0 )*lda + 2*i +  4 ], A[ (j*2 + 0 )*lda + 2*i +  2 ], A[ (j*2 + 0 )*lda + 2*i +  0 ] );
            a1  = _mm256_set_pd( A[ (j*2 + 0 )*lda + 2*i +  7 ], A[ (j*2 + 0 )*lda + 2*i +  5 ], A[ (j*2 + 0 )*lda + 2*i +  3 ], A[ (j*2 + 0 )*lda + 2*i +  1 ] );
            a2  = _mm256_set_pd( A[ (j*2 + 0 )*lda + 2*i +  8 ], A[ (j*2 + 0 )*lda + 2*i +  6 ], A[ (j*2 + 0 )*lda + 2*i +  4 ], A[ (j*2 + 0 )*lda + 2*i +  2 ] );
            a3  = _mm256_set_pd( A[ (j*2 + 0 )*lda + 2*i +  9 ], A[ (j*2 + 0 )*lda + 2*i +  7 ], A[ (j*2 + 0 )*lda + 2*i +  5 ], A[ (j*2 + 0 )*lda + 2*i +  3 ] );

            a4  = _mm256_set_pd( A[ (j*2 + 1 )*lda + 2*i +  6 ], A[ (j*2 + 1 )*lda + 2*i +  4 ], A[ (j*2 + 1 )*lda + 2*i +  2 ], A[ (j*2 + 1 )*lda + 2*i +  0 ] );
            a5  = _mm256_set_pd( A[ (j*2 + 1 )*lda + 2*i +  7 ], A[ (j*2 + 1 )*lda + 2*i +  5 ], A[ (j*2 + 1 )*lda + 2*i +  3 ], A[ (j*2 + 1 )*lda + 2*i +  1 ] );
            a6  = _mm256_set_pd( A[ (j*2 + 1 )*lda + 2*i +  8 ], A[ (j*2 + 1 )*lda + 2*i +  6 ], A[ (j*2 + 1 )*lda + 2*i +  4 ], A[ (j*2 + 1 )*lda + 2*i +  2 ] );
            a7  = _mm256_set_pd( A[ (j*2 + 1 )*lda + 2*i +  9 ], A[ (j*2 + 1 )*lda + 2*i +  7 ], A[ (j*2 + 1 )*lda + 2*i +  5 ], A[ (j*2 + 1 )*lda + 2*i +  3 ] );

            a8  = _mm256_set_pd( A[ (j*2 + 2 )*lda + 2*i +  6 ], A[ (j*2 + 2 )*lda + 2*i +  4 ], A[ (j*2 + 2 )*lda + 2*i +  2 ], A[ (j*2 + 2 )*lda + 2*i +  0 ] );
            a9  = _mm256_set_pd( A[ (j*2 + 2 )*lda + 2*i +  7 ], A[ (j*2 + 2 )*lda + 2*i +  5 ], A[ (j*2 + 2 )*lda + 2*i +  3 ], A[ (j*2 + 2 )*lda + 2*i +  1 ] );
            aa  = _mm256_set_pd( A[ (j*2 + 2 )*lda + 2*i +  8 ], A[ (j*2 + 2 )*lda + 2*i +  6 ], A[ (j*2 + 2 )*lda + 2*i +  4 ], A[ (j*2 + 2 )*lda + 2*i +  2 ] );
            ab  = _mm256_set_pd( A[ (j*2 + 2 )*lda + 2*i +  9 ], A[ (j*2 + 2 )*lda + 2*i +  7 ], A[ (j*2 + 2 )*lda + 2*i +  5 ], A[ (j*2 + 2 )*lda + 2*i +  3 ] );

            ac  = _mm256_set_pd( A[ (j*2 + 3 )*lda + 2*i +  6 ], A[ (j*2 + 3 )*lda + 2*i +  4 ], A[ (j*2 + 3 )*lda + 2*i +  2 ], A[ (j*2 + 3 )*lda + 2*i +  0 ] );
            ad  = _mm256_set_pd( A[ (j*2 + 3 )*lda + 2*i +  7 ], A[ (j*2 + 3 )*lda + 2*i +  5 ], A[ (j*2 + 3 )*lda + 2*i +  3 ], A[ (j*2 + 3 )*lda + 2*i +  1 ] );
            ae  = _mm256_set_pd( A[ (j*2 + 3 )*lda + 2*i +  8 ], A[ (j*2 + 3 )*lda + 2*i +  6 ], A[ (j*2 + 3 )*lda + 2*i +  4 ], A[ (j*2 + 3 )*lda + 2*i +  2 ] );
            af  = _mm256_set_pd( A[ (j*2 + 3 )*lda + 2*i +  9 ], A[ (j*2 + 3 )*lda + 2*i +  7 ], A[ (j*2 + 3 )*lda + 2*i +  5 ], A[ (j*2 + 3 )*lda + 2*i +  3 ] );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );
            w0 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );
            w0m = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a4, ah0,  _mm256_mul_pd( a5, ah1 ) );
            s1 = _mm256_fmadd_pd( a6, ah2,  _mm256_mul_pd( a7, ah3 ) );
            w1 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a4, ag0,  _mm256_mul_pd( a5, ag1 ) );
            s1 = _mm256_fmadd_pd( a6, ag2,  _mm256_mul_pd( a7, ag3 ) );
            w1m = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a8, ah0,  _mm256_mul_pd( a9, ah1 ) );
            s1 = _mm256_fmadd_pd( aa, ah2,  _mm256_mul_pd( ab, ah3 ) );
            w2 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( a8, ag0,  _mm256_mul_pd( a9, ag1 ) );
            s1 = _mm256_fmadd_pd( aa, ag2,  _mm256_mul_pd( ab, ag3 ) );
            w2m = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( ac, ah0,  _mm256_mul_pd( ad, ah1 ) );
            s1 = _mm256_fmadd_pd( ae, ah2,  _mm256_mul_pd( af, ah3 ) );
            w3 = _mm256_add_pd( s0, s1 );

            s0 = _mm256_fmadd_pd( ac, ag0,  _mm256_mul_pd( ad, ag1 ) );
            s1 = _mm256_fmadd_pd( ae, ag2,  _mm256_mul_pd( af, ag3 ) );
            w3m = _mm256_add_pd( s0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
            s1 = _mm256_fmadd_pd( w1, ah1, s0 );
            w = _mm256_fmadd_pd( w0, ah0, s1 );
            _mm256_storeu_pd( &B[ j*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
            s1 = _mm256_fmadd_pd( w1, ag1, s0 );
            w = _mm256_fmadd_pd( w0, ag0, s1 );
            _mm256_storeu_pd( &B[ (j + M/2)*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
            s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
            w = _mm256_fmadd_pd( w0m, ah0, s1 );
            _mm256_storeu_pd( &B[ j*ldb + i + N/2 ], w );

            s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
            s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
            w = _mm256_fmadd_pd( w0m, ag0, s1 );
            _mm256_storeu_pd( &B[ (j + M/2)*ldb + i + N / 2 ], w );
        }

        /* Peeling */

        a0  = _mm256_set_pd( A[ (j*2 + 0 )* lda + N - 2 ] , A[ (j*2 + 0 )* lda + N - 4 ] , A[ (j*2 + 0 )* lda + N - 6 ] , A[ (j*2 + 0 )* lda + N - 8 ]  );
        a1  = _mm256_set_pd( A[ (j*2 + 0 )* lda + N - 1 ] , A[ (j*2 + 0 )* lda + N - 3 ] , A[ (j*2 + 0 )* lda + N - 5 ] , A[ (j*2 + 0 )* lda + N - 7 ]  );
        a2  = _mm256_set_pd( A[ (j*2 + 0 )* lda + 0 ], A[ (j*2 + 0 )* lda + N - 2 ] , A[ (j*2 + 0 )* lda + N - 4 ] , A[ (j*2 + 0 )* lda + N - 6 ]  );
        a3  = _mm256_set_pd( A[ (j*2 + 0 )* lda + 1 ], A[ (j*2 + 0 )* lda + N - 1 ] , A[ (j*2 + 0 )* lda + N - 3 ] , A[ (j*2 + 0 )* lda + N - 5 ]  );

        a4  = _mm256_set_pd( A[ (j*2 + 1 )* lda + N - 2 ] , A[ (j*2 + 1 )* lda + N - 4 ] , A[ (j*2 + 1 )* lda + N - 6 ] , A[ (j*2 + 1 )* lda + N - 8 ]  );
        a5  = _mm256_set_pd( A[ (j*2 + 1 )* lda + N - 1 ] , A[ (j*2 + 1 )* lda + N - 3 ] , A[ (j*2 + 1 )* lda + N - 5 ] , A[ (j*2 + 1 )* lda + N - 7 ]  );
        a6  = _mm256_set_pd( A[ (j*2 + 1 )* lda + 0 ], A[ (j*2 + 1 )* lda + N - 2 ] , A[ (j*2 + 1 )* lda + N - 4 ] , A[ (j*2 + 1 )* lda + N - 6 ]  );
        a7  = _mm256_set_pd( A[ (j*2 + 1 )* lda + 1 ], A[ (j*2 + 1 )* lda + N - 1 ] , A[ (j*2 + 1 )* lda + N - 3 ] , A[ (j*2 + 1 )* lda + N - 5 ]  );

        a8  = _mm256_set_pd( A[ (j*2 + 2 )* lda + N - 2 ] , A[ (j*2 + 2 )* lda + N - 4 ] , A[ (j*2 + 2 )* lda + N - 6 ] , A[ (j*2 + 2 )* lda + N - 8 ]  );
        a9  = _mm256_set_pd( A[ (j*2 + 2 )* lda + N - 1 ] , A[ (j*2 + 2 )* lda + N - 3 ] , A[ (j*2 + 2 )* lda + N - 5 ] , A[ (j*2 + 2 )* lda + N - 7 ]  );
        aa  = _mm256_set_pd( A[ (j*2 + 2 )* lda + 0 ], A[ (j*2 + 2 )* lda + N - 2 ] , A[ (j*2 + 2 )* lda + N - 4 ] , A[ (j*2 + 2 )* lda + N - 6 ]  );
        ab  = _mm256_set_pd( A[ (j*2 + 2 )* lda + 1 ], A[ (j*2 + 2 )* lda + N - 1 ] , A[ (j*2 + 2 )* lda + N - 3 ] , A[ (j*2 + 2 )* lda + N - 5 ]  );

        ac  = _mm256_set_pd( A[ (j*2 + 3 )* lda + N - 2 ] , A[ (j*2 + 3 )* lda + N - 4 ] , A[ (j*2 + 3 )* lda + N - 6 ] , A[ (j*2 + 3 )* lda + N - 8 ]  );
        ad  = _mm256_set_pd( A[ (j*2 + 3 )* lda + N - 1 ] , A[ (j*2 + 3 )* lda + N - 3 ] , A[ (j*2 + 3 )* lda + N - 5 ] , A[ (j*2 + 3 )* lda + N - 7 ]  );
        ae  = _mm256_set_pd( A[ (j*2 + 3 )* lda + 0 ], A[ (j*2 + 3 )* lda + N - 2 ] , A[ (j*2 + 3 )* lda + N - 4 ] , A[ (j*2 + 3 )* lda + N - 6 ]  );
        af  = _mm256_set_pd( A[ (j*2 + 3 )* lda + 1 ], A[ (j*2 + 3 )* lda + N - 1 ] , A[ (j*2 + 3 )* lda + N - 3 ] , A[ (j*2 + 3 )* lda + N - 5 ]  );

        /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

        s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
        s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );
        w0 = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
        s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );
        w0m = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( a4, ah0,  _mm256_mul_pd( a5, ah1 ) );
        s1 = _mm256_fmadd_pd( a6, ah2,  _mm256_mul_pd( a7, ah3 ) );
        w1 = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( a4, ag0,  _mm256_mul_pd( a5, ag1 ) );
        s1 = _mm256_fmadd_pd( a6, ag2,  _mm256_mul_pd( a7, ag3 ) );
        w1m = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( a8, ah0,  _mm256_mul_pd( a9, ah1 ) );
        s1 = _mm256_fmadd_pd( aa, ah2,  _mm256_mul_pd( ab, ah3 ) );
        w2 = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( a8, ag0,  _mm256_mul_pd( a9, ag1 ) );
        s1 = _mm256_fmadd_pd( aa, ag2,  _mm256_mul_pd( ab, ag3 ) );
        w2m = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( ac, ah0,  _mm256_mul_pd( ad, ah1 ) );
        s1 = _mm256_fmadd_pd( ae, ah2,  _mm256_mul_pd( af, ah3 ) );
        w3 = _mm256_add_pd( s0, s1 );

        s0 = _mm256_fmadd_pd( ac, ag0,  _mm256_mul_pd( ad, ag1 ) );
        s1 = _mm256_fmadd_pd( ae, ag2,  _mm256_mul_pd( af, ag3 ) );
        w3m = _mm256_add_pd( s0, s1 );

        /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

        s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
        s1 = _mm256_fmadd_pd( w1, ah1, s0 );
        w = _mm256_fmadd_pd( w0, ah0, s1 );
        _mm256_storeu_pd( &B[ j*ldb + (N/2 - 4) ], w );

        s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
        s1 = _mm256_fmadd_pd( w1, ag1, s0 );
        w = _mm256_fmadd_pd( w0, ag0, s1 );
        _mm256_storeu_pd( &B[ (j + M/2)*ldb + (N/2 - 4) ], w );

        s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
        s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
        w = _mm256_fmadd_pd( w0m, ah0, s1 );
        _mm256_storeu_pd( &B[ j*ldb + N - 4  ], w );

        s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
        s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
        w = _mm256_fmadd_pd( w0m, ag0, s1 );
        _mm256_storeu_pd( &B[ (j + M/2)*ldb + N - 4 ], w );

   }

   /*  Peeling */

   for( i = 0 ; i < N / 2 - 2 ; i+=4 ){

     a0  = _mm256_set_pd( A[ ( M -  2 )* lda  + 2*i +  6 ], A[ ( M -  2 )* lda  + 2*i +  4 ], A[ ( M -  2 )* lda  + 2*i +  2 ], A[ ( M -  2 )* lda  + 2*i +  0 ] );
     a1  = _mm256_set_pd( A[ ( M -  2 )* lda  + 2*i +  7 ], A[ ( M -  2 )* lda  + 2*i +  5 ], A[ ( M -  2 )* lda  + 2*i +  3 ], A[ ( M -  2 )* lda  + 2*i +  1 ] );
     a2  = _mm256_set_pd( A[ ( M -  2 )* lda  + 2*i +  8 ], A[ ( M -  2 )* lda  + 2*i +  6 ], A[ ( M -  2 )* lda  + 2*i +  4 ], A[ ( M -  2 )* lda  + 2*i +  2 ] );
     a3  = _mm256_set_pd( A[ ( M -  2 )* lda  + 2*i +  9 ], A[ ( M -  2 )* lda  + 2*i +  7 ], A[ ( M -  2 )* lda  + 2*i +  5 ], A[ ( M -  2 )* lda  + 2*i +  3 ] );

     a4  = _mm256_set_pd( A[ ( M -  1 )* lda  + 2*i +  6 ], A[ ( M -  1 )* lda  + 2*i +  4 ], A[ ( M -  1 )* lda  + 2*i +  2 ], A[ ( M -  1 )* lda  + 2*i +  0 ] );
     a5  = _mm256_set_pd( A[ ( M -  1 )* lda  + 2*i +  7 ], A[ ( M -  1 )* lda  + 2*i +  5 ], A[ ( M -  1 )* lda  + 2*i +  3 ], A[ ( M -  1 )* lda  + 2*i +  1 ] );
     a6  = _mm256_set_pd( A[ ( M -  1 )* lda  + 2*i +  8 ], A[ ( M -  1 )* lda  + 2*i +  6 ], A[ ( M -  1 )* lda  + 2*i +  4 ], A[ ( M -  1 )* lda  + 2*i +  2 ] );
     a7  = _mm256_set_pd( A[ ( M -  1 )* lda  + 2*i +  9 ], A[ ( M -  1 )* lda  + 2*i +  7 ], A[ ( M -  1 )* lda  + 2*i +  5 ], A[ ( M -  1 )* lda  + 2*i +  3 ] );

     a8  = _mm256_set_pd( A[ ( 0 )* lda  + 2*i +  6 ], A[ ( 0 )* lda  + 2*i +  4 ], A[ ( 0 )* lda  + 2*i +  2 ], A[ ( 0 )* lda  + 2*i +  0 ] );
     a9  = _mm256_set_pd( A[ ( 0 )* lda  + 2*i +  7 ], A[ ( 0 )* lda  + 2*i +  5 ], A[ ( 0 )* lda  + 2*i +  3 ], A[ ( 0 )* lda  + 2*i +  1 ] );
     aa  = _mm256_set_pd( A[ ( 0 )* lda  + 2*i +  8 ], A[ ( 0 )* lda  + 2*i +  6 ], A[ ( 0 )* lda  + 2*i +  4 ], A[ ( 0 )* lda  + 2*i +  2 ] );
     ab  = _mm256_set_pd( A[ ( 0 )* lda  + 2*i +  9 ], A[ ( 0 )* lda  + 2*i +  7 ], A[ ( 0 )* lda  + 2*i +  5 ], A[ ( 0 )* lda  + 2*i +  3 ] );

     ac  = _mm256_set_pd( A[ ( 1 )* lda  + 2*i +  6 ], A[ ( 1 )* lda  + 2*i +  4 ], A[ ( 1 )* lda  + 2*i +  2 ], A[ ( 1 )* lda  + 2*i +  0 ] );
     ad  = _mm256_set_pd( A[ ( 1 )* lda  + 2*i +  7 ], A[ ( 1 )* lda  + 2*i +  5 ], A[ ( 1 )* lda  + 2*i +  3 ], A[ ( 1 )* lda  + 2*i +  1 ] );
     ae  = _mm256_set_pd( A[ ( 1 )* lda  + 2*i +  8 ], A[ ( 1 )* lda  + 2*i +  6 ], A[ ( 1 )* lda  + 2*i +  4 ], A[ ( 1 )* lda  + 2*i +  2 ] );
     af  = _mm256_set_pd( A[ ( 1 )* lda  + 2*i +  9 ], A[ ( 1 )* lda  + 2*i +  7 ], A[ ( 1 )* lda  + 2*i +  5 ], A[ ( 1 )* lda  + 2*i +  3 ] );

     /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

     s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
     s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );
     w0 = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
     s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );
     w0m = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( a4, ah0,  _mm256_mul_pd( a5, ah1 ) );
     s1 = _mm256_fmadd_pd( a6, ah2,  _mm256_mul_pd( a7, ah3 ) );
     w1 = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( a4, ag0,  _mm256_mul_pd( a5, ag1 ) );
     s1 = _mm256_fmadd_pd( a6, ag2,  _mm256_mul_pd( a7, ag3 ) );
     w1m = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( a8, ah0,  _mm256_mul_pd( a9, ah1 ) );
     s1 = _mm256_fmadd_pd( aa, ah2,  _mm256_mul_pd( ab, ah3 ) );
     w2 = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( a8, ag0,  _mm256_mul_pd( a9, ag1 ) );
     s1 = _mm256_fmadd_pd( aa, ag2,  _mm256_mul_pd( ab, ag3 ) );
     w2m = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( ac, ah0,  _mm256_mul_pd( ad, ah1 ) );
     s1 = _mm256_fmadd_pd( ae, ah2,  _mm256_mul_pd( af, ah3 ) );
     w3 = _mm256_add_pd( s0, s1 );

     s0 = _mm256_fmadd_pd( ac, ag0,  _mm256_mul_pd( ad, ag1 ) );
     s1 = _mm256_fmadd_pd( ae, ag2,  _mm256_mul_pd( af, ag3 ) );
     w3m = _mm256_add_pd( s0, s1 );

     /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

     s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
     s1 = _mm256_fmadd_pd( w1, ah1, s0 );
     w = _mm256_fmadd_pd( w0, ah0, s1 );
     _mm256_storeu_pd( &B[ (M/2 - 1)*ldb + i ], w );

     s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
     s1 = _mm256_fmadd_pd( w1, ag1, s0 );
     w = _mm256_fmadd_pd( w0, ag0, s1 );
     _mm256_storeu_pd( &B[ (M - 1)*ldb + i ], w );

     s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
     s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
     w = _mm256_fmadd_pd( w0m, ah0, s1 );
     _mm256_storeu_pd( &B[ (M/2 - 1)*ldb + i + N/2 ], w );

     s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
     s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
     w = _mm256_fmadd_pd( w0m, ag0, s1 );
     _mm256_storeu_pd( &B[ (M - 1)*ldb + i + N / 2 ], w );

   }

   /* Peeling: last corner */

   a0  = _mm256_set_pd( A[ ( M -  2 )* lda + N - 2 ] , A[ ( M -  2 )* lda + N - 4 ] , A[ ( M -  2 )* lda + N - 6 ] , A[ ( M -  2 )* lda + N - 8 ]  );
   a1  = _mm256_set_pd( A[ ( M -  2 )* lda + N - 1 ] , A[ ( M -  2 )* lda + N - 3 ] , A[ ( M -  2 )* lda + N - 5 ] , A[ ( M -  2 )* lda + N - 7 ]  );
   a2  = _mm256_set_pd( A[ ( M -  2 )* lda + 0 ], A[ ( M -  2 )* lda + N - 2 ] , A[ ( M -  2 )* lda + N - 4 ] , A[ ( M -  2 )* lda + N - 6 ]  );
   a3  = _mm256_set_pd( A[ ( M -  2 )* lda + 1 ], A[ ( M -  2 )* lda + N - 1 ] , A[ ( M -  2 )* lda + N - 3 ] , A[ ( M -  2 )* lda + N - 5 ]  );

   a4  = _mm256_set_pd( A[ ( M -  1 )* lda + N - 2 ] , A[ ( M -  1 )* lda + N - 4 ] , A[ ( M -  1 )* lda + N - 6 ] , A[ ( M -  1 )* lda + N - 8 ]  );
   a5  = _mm256_set_pd( A[ ( M -  1 )* lda + N - 1 ] , A[ ( M -  1 )* lda + N - 3 ] , A[ ( M -  1 )* lda + N - 5 ] , A[ ( M -  1 )* lda + N - 7 ]  );
   a6  = _mm256_set_pd( A[ ( M -  1 )* lda + 0 ], A[ ( M -  1 )* lda + N - 2 ] , A[ ( M -  1 )* lda + N - 4 ] , A[ ( M -  1 )* lda + N - 6 ]  );
   a7  = _mm256_set_pd( A[ ( M -  1 )* lda + 1 ], A[ ( M -  1 )* lda + N - 1 ] , A[ ( M -  1 )* lda + N - 3 ] , A[ ( M -  1 )* lda + N - 5 ]  );

   a8  = _mm256_set_pd( A[ ( 0 )* lda + N - 2 ] , A[ ( 0 )* lda + N - 4 ] , A[ ( 0 )* lda + N - 6 ] , A[ ( 0 )* lda + N - 8 ]  );
   a9  = _mm256_set_pd( A[ ( 0 )* lda + N - 1 ] , A[ ( 0 )* lda + N - 3 ] , A[ ( 0 )* lda + N - 5 ] , A[ ( 0 )* lda + N - 7 ]  );
   aa  = _mm256_set_pd( A[ ( 0 )* lda + 0 ], A[ ( 0 )* lda + N - 2 ] , A[ ( 0 )* lda + N - 4 ] , A[ ( 0 )* lda + N - 6 ]  );
   ab  = _mm256_set_pd( A[ ( 0 )* lda + 1 ], A[ ( 0 )* lda + N - 1 ] , A[ ( 0 )* lda + N - 3 ] , A[ ( 0 )* lda + N - 5 ]  );

   ac  = _mm256_set_pd( A[ ( 1 )* lda + N - 2 ] , A[ ( 1 )* lda + N - 4 ] , A[ ( 1 )* lda + N - 6 ] , A[ ( 1 )* lda + N - 8 ]  );
   ad  = _mm256_set_pd( A[ ( 1 )* lda + N - 1 ] , A[ ( 1 )* lda + N - 3 ] , A[ ( 1 )* lda + N - 5 ] , A[ ( 1 )* lda + N - 7 ]  );
   ae  = _mm256_set_pd( A[ ( 1 )* lda + 0 ], A[ ( 1 )* lda + N - 2 ] , A[ ( 1 )* lda + N - 4 ] , A[ ( 1 )* lda + N - 6 ]  );
   af  = _mm256_set_pd( A[ ( 1 )* lda + 1 ], A[ ( 1 )* lda + N - 1 ] , A[ ( 1 )* lda + N - 3 ] , A[ ( 1 )* lda + N - 5 ]  );

   /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

   s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ah1 ) );
   s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ah3 ) );
   w0 = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( a0, ag0,  _mm256_mul_pd( a1, ag1 ) );
   s1 = _mm256_fmadd_pd( a2, ag2,  _mm256_mul_pd( a3, ag3 ) );
   w0m = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( a4, ah0,  _mm256_mul_pd( a5, ah1 ) );
   s1 = _mm256_fmadd_pd( a6, ah2,  _mm256_mul_pd( a7, ah3 ) );
   w1 = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( a4, ag0,  _mm256_mul_pd( a5, ag1 ) );
   s1 = _mm256_fmadd_pd( a6, ag2,  _mm256_mul_pd( a7, ag3 ) );
   w1m = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( a8, ah0,  _mm256_mul_pd( a9, ah1 ) );
   s1 = _mm256_fmadd_pd( aa, ah2,  _mm256_mul_pd( ab, ah3 ) );
   w2 = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( a8, ag0,  _mm256_mul_pd( a9, ag1 ) );
   s1 = _mm256_fmadd_pd( aa, ag2,  _mm256_mul_pd( ab, ag3 ) );
   w2m = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( ac, ah0,  _mm256_mul_pd( ad, ah1 ) );
   s1 = _mm256_fmadd_pd( ae, ah2,  _mm256_mul_pd( af, ah3 ) );
   w3 = _mm256_add_pd( s0, s1 );

   s0 = _mm256_fmadd_pd( ac, ag0,  _mm256_mul_pd( ad, ag1 ) );
   s1 = _mm256_fmadd_pd( ae, ag2,  _mm256_mul_pd( af, ag3 ) );
   w3m = _mm256_add_pd( s0, s1 );

   /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

   s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
   s1 = _mm256_fmadd_pd( w1, ah1, s0 );
   w = _mm256_fmadd_pd( w0, ah0, s1 );
   _mm256_storeu_pd( &B[ (M/2 - 1)*ldb + N/2 - 4 ], w );

   s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
   s1 = _mm256_fmadd_pd( w1, ag1, s0 );
   w = _mm256_fmadd_pd( w0, ag0, s1 );
   _mm256_storeu_pd( &B[ (M - 1)*ldb + N/2 - 4 ], w );

   s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
   s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
   w = _mm256_fmadd_pd( w0m, ah0, s1 );
   _mm256_storeu_pd( &B[ (M/2 - 1)*ldb + N - 4 ], w );

   s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
   s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
   w = _mm256_fmadd_pd( w0m, ag0, s1 );
   _mm256_storeu_pd( &B[ (M - 1)*ldb + N - 4 ], w );
}

#endif // __AVX2__

#ifdef __AVX512F__

void dda4mt2_fma512_reuse( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m512d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m512d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m512d ah0, ah1, ah2, ah3;
    __m512d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm512_set1_pd( h0 );
    ah1 = _mm512_set1_pd( h1 );
    ah2 = _mm512_set1_pd( h2 );
    ah3 = _mm512_set1_pd( h3 );
    ag0 = _mm512_set1_pd( g0 );
    ag1 = _mm512_set1_pd( g1 );
    ag2 = _mm512_set1_pd( g2 );
    ag3 = _mm512_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M / 2; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=8 ){

	  a0 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 14 ) % N],  A[ (j*2)*lda + ( 2*i + 12 ) % N], A[(j*2)*lda + ( 2*i + 10 ) % N ], A[(j*2)*lda + ( 2*i + 8 ) % N ], A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + ( 2*i + 4 ) % N], A[(j*2)*lda + 2*i + 2 ], A[(j*2)*lda + 2*i] );
	  a1 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 15 ) % N],  A[ (j*2)*lda + ( 2*i + 13 ) % N],  A[ (j*2)*lda + ( 2*i + 11 ) % N ],   A[ (j*2)*lda + ( 2*i + 9 ) % N ],  A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + ( 2*i + 5 ) % N],  A[ (j*2)*lda + 2*i + 3 ],   A[ (j*2)*lda + 2*i + 1] );
	  a2 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 16 ) % N],  A[ (j*2)*lda + ( 2*i + 14 ) % N],  A[ (j*2)*lda + ( 2*i + 12 ) % N ],   A[ (j*2)*lda + ( 2*i + 10 ) % N ],  A[ (j*2)*lda + ( 2*i + 8 ) % N],  A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + 2*i + 4 ],   A[ (j*2)*lda + 2*i + 2 ]);
	  a3 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 17 ) % N],  A[ (j*2)*lda + ( 2*i + 15 ) % N],  A[ (j*2)*lda + ( 2*i + 13 ) % N ],   A[ (j*2)*lda + ( 2*i + 11 ) % N ], A[ (j*2)*lda + ( 2*i + 9 ) % N],  A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + 2*i + 5 ],   A[ (j*2)*lda + 2*i + 3 ] );

	  a4 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 14 ) % N],  A[ (j*2+1)*lda + ( 2*i + 12 ) % N], A[(j*2+1)*lda + ( 2*i + 10 ) % N ], A[(j*2+1)*lda + ( 2*i + 8 ) % N ], A[(j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + ( 2*i + 4 ) % N], A[(j*2+1)*lda + 2*i + 2 ], A[(j*2+1)*lda + 2*i] );
	  a5 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 15 ) % N],  A[ (j*2+1)*lda + ( 2*i + 13 ) % N],  A[ (j*2+1)*lda + ( 2*i + 11 ) % N ],   A[ (j*2+1)*lda + ( 2*i + 9 ) % N ],  A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + ( 2*i + 5 ) % N],  A[ (j*2+1)*lda + 2*i + 3 ],   A[ (j*2+1)*lda + 2*i + 1] );
	  a6 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 16 ) % N],  A[ (j*2+1)*lda + ( 2*i + 14 ) % N],  A[ (j*2+1)*lda + ( 2*i + 12 ) % N ],   A[ (j*2+1)*lda + ( 2*i + 10 ) % N ], A[ (j*2+1)*lda + ( 2*i + 8 ) % N],  A[ (j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + 2*i + 4 ],   A[ (j*2+1)*lda + 2*i + 2 ]);
	  a7 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 17 ) % N],  A[ (j*2+1)*lda + ( 2*i + 15 ) % N],  A[ (j*2+1)*lda + ( 2*i + 13 ) % N ],   A[ (j*2+1)*lda + ( 2*i + 11 ) % N ],  A[ (j*2+1)*lda + ( 2*i + 9 ) % N],  A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + 2*i + 5 ],   A[ (j*2+1)*lda + 2*i + 3 ] );

	  a8 = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 14 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 12 ) % N], A[((j*2+2)%M)*lda + ( 2*i + 10 ) % N ], A[((j*2+2)%M)*lda + ( 2*i + 8 ) % N ], A[((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 4 ) % N], A[((j*2+2)%M)*lda + 2*i + 2 ], A[((j*2+2)%M)*lda + 2*i] );
	  a9 = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 15 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 13 ) % N], A[((j*2+2)%M)*lda + ( 2*i + 11 ) % N ], A[((j*2+2)%M)*lda + ( 2*i + 9 ) % N ], A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 5 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 3 ],   A[ ((j*2+2)%M)*lda + 2*i + 1] );
	  aa = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 16 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 14 ) % N], A[((j*2+2)%M)*lda + ( 2*i + 12 ) % N ], A[((j*2+2)%M)*lda + ( 2*i + 10 ) % N ], A[ ((j*2+2)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 4 ],   A[ ((j*2+2)%M)*lda + 2*i + 2 ]);
	  ab = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 17 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 15 ) % N], A[((j*2+2)%M)*lda + ( 2*i + 13 ) % N ], A[((j*2+2)%M)*lda + ( 2*i + 11 ) % N ], A[ ((j*2+2)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 5 ],   A[ ((j*2+2)%M)*lda + 2*i + 3 ] );

	  ac = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 14 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 12 ) % N], A[((j*2+3)%M)*lda + ( 2*i + 10 ) % N ], A[((j*2+3)%M)*lda + ( 2*i + 8 ) % N ], A[((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 4 ) % N], A[((j*2+3)%M)*lda + 2*i + 2 ], A[((j*2+3)%M)*lda + 2*i] );
	  ad = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 15 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 13 ) % N], A[((j*2+3)%M)*lda + ( 2*i + 11 ) % N ], A[((j*2+3)%M)*lda + ( 2*i + 9 ) % N ], A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 5 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 3 ],   A[ ((j*2+3)%M)*lda + 2*i + 1] );
	  ae = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 16 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 14 ) % N], A[((j*2+3)%M)*lda + ( 2*i + 12 ) % N ], A[((j*2+3)%M)*lda + ( 2*i + 10 ) % N ], A[ ((j*2+3)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 4 ],   A[ ((j*2+3)%M)*lda + 2*i + 2 ]);
	  af = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 17 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 15 ) % N], A[((j*2+3)%M)*lda + ( 2*i + 13 ) % N ], A[((j*2+3)%M)*lda + ( 2*i + 11 ) % N ], A[ ((j*2+3)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 5 ],   A[ ((j*2+3)%M)*lda + 2*i + 3 ] );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm512_fmadd_pd( a0, ah0,  _mm512_mul_pd( a1, ah1 ) );
            s1 = _mm512_fmadd_pd( a2, ah2,  _mm512_mul_pd( a3, ah3 ) );
            w0 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a0, ag0,  _mm512_mul_pd( a1, ag1 ) );
            s1 = _mm512_fmadd_pd( a2, ag2,  _mm512_mul_pd( a3, ag3 ) );
            w0m = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a4, ah0,  _mm512_mul_pd( a5, ah1 ) );
            s1 = _mm512_fmadd_pd( a6, ah2,  _mm512_mul_pd( a7, ah3 ) );
            w1 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a4, ag0,  _mm512_mul_pd( a5, ag1 ) );
            s1 = _mm512_fmadd_pd( a6, ag2,  _mm512_mul_pd( a7, ag3 ) );
            w1m = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a8, ah0,  _mm512_mul_pd( a9, ah1 ) );
            s1 = _mm512_fmadd_pd( aa, ah2,  _mm512_mul_pd( ab, ah3 ) );
            w2 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a8, ag0,  _mm512_mul_pd( a9, ag1 ) );
            s1 = _mm512_fmadd_pd( aa, ag2,  _mm512_mul_pd( ab, ag3 ) );
            w2m = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( ac, ah0,  _mm512_mul_pd( ad, ah1 ) );
            s1 = _mm512_fmadd_pd( ae, ah2,  _mm512_mul_pd( af, ah3 ) );
            w3 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( ac, ag0,  _mm512_mul_pd( ad, ag1 ) );
            s1 = _mm512_fmadd_pd( ae, ag2,  _mm512_mul_pd( af, ag3 ) );
            w3m = _mm512_add_pd( s0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm512_fmadd_pd( w2, ah2, _mm512_mul_pd( w3, ah3 ) );
            s1 = _mm512_fmadd_pd( w1, ah1, s0 );
            w = _mm512_fmadd_pd( w0, ah0, s1 );
            _mm512_storeu_pd( &B[ 1*j*ldb + i ], w );

            s0 = _mm512_fmadd_pd( w2, ag2, _mm512_mul_pd( w3, ag3 ) );
            s1 = _mm512_fmadd_pd( w1, ag1, s0 );
            w = _mm512_fmadd_pd( w0, ag0, s1 );
            _mm512_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

            s0 = _mm512_fmadd_pd( w2m, ah2, _mm512_mul_pd( w3m, ah3 ) );
            s1 = _mm512_fmadd_pd( w1m, ah1, s0 );
            w = _mm512_fmadd_pd( w0m, ah0, s1 );
            _mm512_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

            s0 = _mm512_fmadd_pd( w2m, ag2, _mm512_mul_pd( w3m, ag3 ) );
            s1 = _mm512_fmadd_pd( w1m, ag1, s0 );
            w = _mm512_fmadd_pd( w0m, ag0, s1 );
            _mm512_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );

        }
   }

}

/* TODO */

 void dda4mt2_fma512_reuse_gather( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m512d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m512d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m512d ah0, ah1, ah2, ah3;
    __m512d ag0, ag1, ag2, ag3;
    __m512i stride = _mm512_set_epi64( (long long int) 7*sizeof( double ), (long long int) 6*sizeof( double ),
				       (long long int) 5*sizeof( double ), (long long int) 4*sizeof( double ),
				       (long long int) 3*sizeof( double ), (long long int) 2*sizeof( double ),
				       (long long int)   sizeof( double ), (long long int) 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm512_set1_pd( h0 );
    ah1 = _mm512_set1_pd( h1 );
    ah2 = _mm512_set1_pd( h2 );
    ah3 = _mm512_set1_pd( h3 );
    ag0 = _mm512_set1_pd( g0 );
    ag1 = _mm512_set1_pd( g1 );
    ag2 = _mm512_set1_pd( g2 );
    ag3 = _mm512_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M / 2; j++ ) {
        for( i = 0 ; i < N / 2 - 8 ; i+=8 ){

	  a0 = _mm512_i64gather_pd( stride, &A[(j*2)*lda + 2*i], 2 );
	  a1 = _mm512_i64gather_pd( stride, &A[(j*2)*lda + 2*i + 1], 2 );
	  a2 = _mm512_i64gather_pd( stride, &A[(j*2)*lda + 2*i + 2], 2 );
	  a3 = _mm512_i64gather_pd( stride, &A[(j*2)*lda + 2*i + 3], 2 );

	  a4 = _mm512_i64gather_pd( stride, &A[(j*2+1)*lda + 2*i], 2 );
	  a5 = _mm512_i64gather_pd( stride, &A[(j*2+1)*lda + 2*i + 1], 2 );
	  a6 = _mm512_i64gather_pd( stride, &A[(j*2+1)*lda + 2*i + 2], 2 );
	  a7 = _mm512_i64gather_pd( stride, &A[(j*2+1)*lda + 2*i + 3], 2 );

	  a8 = _mm512_i64gather_pd( stride, &A[((j*2+2)%M)*lda + 2*i], 2 );
	  a9 = _mm512_i64gather_pd( stride, &A[((j*2+2)%M)*lda + 2*i + 1], 2 );
	  aa = _mm512_i64gather_pd( stride, &A[((j*2+2)%M)*lda + 2*i + 2], 2 );
	  ab = _mm512_i64gather_pd( stride, &A[((j*2+2)%M)*lda + 2*i + 3], 2 );

	  ac = _mm512_i64gather_pd( stride, &A[((j*2+3)%M)*lda + 2*i], 2 );
	  ad = _mm512_i64gather_pd( stride, &A[((j*2+3)%M)*lda + 2*i + 1], 2 );
	  ae = _mm512_i64gather_pd( stride, &A[((j*2+3)%M)*lda + 2*i + 2], 2 );
	  af = _mm512_i64gather_pd( stride, &A[((j*2+3)%M)*lda + 2*i + 3], 2 );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm512_fmadd_pd( a0, ah0,  _mm512_mul_pd( a1, ah1 ) );
            s1 = _mm512_fmadd_pd( a2, ah2,  _mm512_mul_pd( a3, ah3 ) );
            w0 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a0, ag0,  _mm512_mul_pd( a1, ag1 ) );
            s1 = _mm512_fmadd_pd( a2, ag2,  _mm512_mul_pd( a3, ag3 ) );
            w0m = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a4, ah0,  _mm512_mul_pd( a5, ah1 ) );
            s1 = _mm512_fmadd_pd( a6, ah2,  _mm512_mul_pd( a7, ah3 ) );
            w1 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a4, ag0,  _mm512_mul_pd( a5, ag1 ) );
            s1 = _mm512_fmadd_pd( a6, ag2,  _mm512_mul_pd( a7, ag3 ) );
            w1m = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a8, ah0,  _mm512_mul_pd( a9, ah1 ) );
            s1 = _mm512_fmadd_pd( aa, ah2,  _mm512_mul_pd( ab, ah3 ) );
            w2 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( a8, ag0,  _mm512_mul_pd( a9, ag1 ) );
            s1 = _mm512_fmadd_pd( aa, ag2,  _mm512_mul_pd( ab, ag3 ) );
            w2m = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( ac, ah0,  _mm512_mul_pd( ad, ah1 ) );
            s1 = _mm512_fmadd_pd( ae, ah2,  _mm512_mul_pd( af, ah3 ) );
            w3 = _mm512_add_pd( s0, s1 );

            s0 = _mm512_fmadd_pd( ac, ag0,  _mm512_mul_pd( ad, ag1 ) );
            s1 = _mm512_fmadd_pd( ae, ag2,  _mm512_mul_pd( af, ag3 ) );
            w3m = _mm512_add_pd( s0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm512_fmadd_pd( w2, ah2, _mm512_mul_pd( w3, ah3 ) );
            s1 = _mm512_fmadd_pd( w1, ah1, s0 );
            w = _mm512_fmadd_pd( w0, ah0, s1 );
            _mm512_storeu_pd( &B[ 1*j*ldb + i ], w );

            s0 = _mm512_fmadd_pd( w2, ag2, _mm512_mul_pd( w3, ag3 ) );
            s1 = _mm512_fmadd_pd( w1, ag1, s0 );
            w = _mm512_fmadd_pd( w0, ag0, s1 );
            _mm512_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

            s0 = _mm512_fmadd_pd( w2m, ah2, _mm512_mul_pd( w3m, ah3 ) );
            s1 = _mm512_fmadd_pd( w1m, ah1, s0 );
            w = _mm512_fmadd_pd( w0m, ah0, s1 );
            _mm512_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

            s0 = _mm512_fmadd_pd( w2m, ag2, _mm512_mul_pd( w3m, ag3 ) );
            s1 = _mm512_fmadd_pd( w1m, ag1, s0 );
            w = _mm512_fmadd_pd( w0m, ag0, s1 );
            _mm512_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );

        }

	i = N/2 - 8;

	/* The last column cannot be done with gather because of the folding */

	a0 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 14 )],  A[ (j*2)*lda + ( 2*i + 12 )], A[(j*2)*lda + ( 2*i + 10 ) ], A[(j*2)*lda + ( 2*i + 8 ) ], A[ (j*2)*lda + ( 2*i + 6 ) ],  A[ (j*2)*lda + ( 2*i + 4 ) ], A[(j*2)*lda + 2*i + 2 ], A[(j*2)*lda + 2*i] );
	a1 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 15 )],  A[ (j*2)*lda + ( 2*i + 13 ) ],  A[ (j*2)*lda + ( 2*i + 11 )  ],   A[ (j*2)*lda + ( 2*i + 9 )  ],  A[ (j*2)*lda + ( 2*i + 7 ) ],  A[ (j*2)*lda + ( 2*i + 5 ) ],  A[ (j*2)*lda + 2*i + 3 ],   A[ (j*2)*lda + 2*i + 1] );
	a2 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 16 - N)],  A[ (j*2)*lda + ( 2*i + 14 ) ],  A[ (j*2)*lda + ( 2*i + 12 )  ],   A[ (j*2)*lda + ( 2*i + 10 )  ],  A[ (j*2)*lda + ( 2*i + 8 ) ],  A[ (j*2)*lda + ( 2*i + 6 ) ],  A[ (j*2)*lda + 2*i + 4 ],   A[ (j*2)*lda + 2*i + 2 ]);
	a3 = _mm512_set_pd( A[ (j*2)*lda + ( 2*i + 17 - N ) ],  A[ (j*2)*lda + ( 2*i + 15 ) ],  A[ (j*2)*lda + ( 2*i + 13 )  ],   A[ (j*2)*lda + ( 2*i + 11 )  ], A[ (j*2)*lda + ( 2*i + 9 ) ],  A[ (j*2)*lda + ( 2*i + 7 ) ],  A[ (j*2)*lda + 2*i + 5 ],   A[ (j*2)*lda + 2*i + 3 ] );

	a4 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 14 ) ],  A[ (j*2+1)*lda + ( 2*i + 12 ) ], A[(j*2+1)*lda + ( 2*i + 10 )  ], A[(j*2+1)*lda + ( 2*i + 8 )  ], A[(j*2+1)*lda + ( 2*i + 6 ) ],  A[ (j*2+1)*lda + ( 2*i + 4 ) ], A[(j*2+1)*lda + 2*i + 2 ], A[(j*2+1)*lda + 2*i] );
	a5 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 15 ) ],  A[ (j*2+1)*lda + ( 2*i + 13 ) ],  A[ (j*2+1)*lda + ( 2*i + 11 )  ],   A[ (j*2+1)*lda + ( 2*i + 9 )  ],  A[ (j*2+1)*lda + ( 2*i + 7 ) ],  A[ (j*2+1)*lda + ( 2*i + 5 ) ],  A[ (j*2+1)*lda + 2*i + 3 ],   A[ (j*2+1)*lda + 2*i + 1] );
	a6 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 16 - N ) ],  A[ (j*2+1)*lda + ( 2*i + 14 ) ],  A[ (j*2+1)*lda + ( 2*i + 12 )  ],   A[ (j*2+1)*lda + ( 2*i + 10 )  ], A[ (j*2+1)*lda + ( 2*i + 8 ) ],  A[ (j*2+1)*lda + ( 2*i + 6 ) ],  A[ (j*2+1)*lda + 2*i + 4 ],   A[ (j*2+1)*lda + 2*i + 2 ]);
	a7 = _mm512_set_pd( A[ (j*2+1)*lda + ( 2*i + 17 - N ) ],  A[ (j*2+1)*lda + ( 2*i + 15 ) ],  A[ (j*2+1)*lda + ( 2*i + 13 )  ],   A[ (j*2+1)*lda + ( 2*i + 11 )  ],  A[ (j*2+1)*lda + ( 2*i + 9 ) ],  A[ (j*2+1)*lda + ( 2*i + 7 ) ],  A[ (j*2+1)*lda + 2*i + 5 ],   A[ (j*2+1)*lda + 2*i + 3 ] );

	a8 = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 14 ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 12 ) ], A[((j*2+2)%M)*lda + ( 2*i + 10 )  ], A[((j*2+2)%M)*lda + ( 2*i + 8 )  ], A[((j*2+2)%M)*lda + ( 2*i + 6 ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 4 ) ], A[((j*2+2)%M)*lda + 2*i + 2 ], A[((j*2+2)%M)*lda + 2*i] );
	a9 = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 15 ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 13 ) ], A[((j*2+2)%M)*lda + ( 2*i + 11 )  ], A[((j*2+2)%M)*lda + ( 2*i + 9 )  ], A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 5 ) ],  A[ ((j*2+2)%M)*lda + 2*i + 3 ],   A[ ((j*2+2)%M)*lda + 2*i + 1] );
	aa = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 16 - N ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 14 ) ], A[((j*2+2)%M)*lda + ( 2*i + 12 )  ], A[((j*2+2)%M)*lda + ( 2*i + 10 )  ], A[ ((j*2+2)%M)*lda + ( 2*i + 8 ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) ],  A[ ((j*2+2)%M)*lda + 2*i + 4 ],   A[ ((j*2+2)%M)*lda + 2*i + 2 ]);
	ab = _mm512_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 17 - N ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 15 ) ], A[((j*2+2)%M)*lda + ( 2*i + 13 )  ], A[((j*2+2)%M)*lda + ( 2*i + 11 )  ], A[ ((j*2+2)%M)*lda + ( 2*i + 9 ) ],  A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) ],  A[ ((j*2+2)%M)*lda + 2*i + 5 ],   A[ ((j*2+2)%M)*lda + 2*i + 3 ] );

	ac = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 14 ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 12 ) ], A[((j*2+3)%M)*lda + ( 2*i + 10 )  ], A[((j*2+3)%M)*lda + ( 2*i + 8 )  ], A[((j*2+3)%M)*lda + ( 2*i + 6 ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 4 ) ], A[((j*2+3)%M)*lda + 2*i + 2 ], A[((j*2+3)%M)*lda + 2*i] );
	  ad = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 15 ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 13 ) ], A[((j*2+3)%M)*lda + ( 2*i + 11 )  ], A[((j*2+3)%M)*lda + ( 2*i + 9 )  ], A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 5 ) ],  A[ ((j*2+3)%M)*lda + 2*i + 3 ],   A[ ((j*2+3)%M)*lda + 2*i + 1] );
	  ae = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 16 - N ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 14 ) ], A[((j*2+3)%M)*lda + ( 2*i + 12 )  ], A[((j*2+3)%M)*lda + ( 2*i + 10 )  ], A[ ((j*2+3)%M)*lda + ( 2*i + 8 ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) ],  A[ ((j*2+3)%M)*lda + 2*i + 4 ],   A[ ((j*2+3)%M)*lda + 2*i + 2 ]);
	  af = _mm512_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 17 - N ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 15 ) ], A[((j*2+3)%M)*lda + ( 2*i + 13 )  ], A[((j*2+3)%M)*lda + ( 2*i + 11 )  ], A[ ((j*2+3)%M)*lda + ( 2*i + 9 ) ],  A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) ],  A[ ((j*2+3)%M)*lda + 2*i + 5 ],   A[ ((j*2+3)%M)*lda + 2*i + 3 ] );


        /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

        s0 = _mm512_fmadd_pd( a2, ah2, _mm512_mul_pd( a3, ah3 ) );
        s1 = _mm512_fmadd_pd( a1, ah1, s0 );
        w0 = _mm512_fmadd_pd( a0, ah0, s1 );

        s0 = _mm512_fmadd_pd( a2, ag2, _mm512_mul_pd( a3, ag3 ) );
        s1 = _mm512_fmadd_pd( a1, ag1, s0 );
        w0m = _mm512_fmadd_pd( a0, ag0, s1 );

        s0 = _mm512_fmadd_pd( a6, ah2, _mm512_mul_pd( a7, ah3 ) );
        s1 = _mm512_fmadd_pd( a5, ah1, s0 );
        w1 = _mm512_fmadd_pd( a4, ah0, s1 );

        s0 = _mm512_fmadd_pd( a6, ag2, _mm512_mul_pd( a7, ag3 ) );
        s1 = _mm512_fmadd_pd( a5, ag1, s0 );
        w1m = _mm512_fmadd_pd( a4, ag0, s1 );

        s0 = _mm512_fmadd_pd( aa, ah2, _mm512_mul_pd( ab, ah3 ) );
        s1 = _mm512_fmadd_pd( a9, ah1, s0 );
        w2 = _mm512_fmadd_pd( a8, ah0, s1 );

        s0 = _mm512_fmadd_pd( aa, ag2, _mm512_mul_pd( ab, ag3 ) );
        s1 = _mm512_fmadd_pd( a9, ag1, s0 );
        w2m = _mm512_fmadd_pd( a8, ag0, s1 );

        s0 = _mm512_fmadd_pd( ae, ah2, _mm512_mul_pd( af, ah3 ) );
        s1 = _mm512_fmadd_pd( ad, ah1, s0 );
        w3 = _mm512_fmadd_pd( ac, ah0, s1 );

        s0 = _mm512_fmadd_pd( ae, ag2, _mm512_mul_pd( af, ag3 ) );
        s1 = _mm512_fmadd_pd( ad, ag1, s0 );
        w3m = _mm512_fmadd_pd( ac, ag0, s1 );

        /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

        s0 = _mm512_fmadd_pd( w2, ah2, _mm512_mul_pd( w3, ah3 ) );
        s1 = _mm512_fmadd_pd( w1, ah1, s0 );
        w = _mm512_fmadd_pd( w0, ah0, s1 );
        _mm512_storeu_pd( &B[ 1*j*ldb + i ], w );

        s0 = _mm512_fmadd_pd( w2, ag2, _mm512_mul_pd( w3, ag3 ) );
        s1 = _mm512_fmadd_pd( w1, ag1, s0 );
        w = _mm512_fmadd_pd( w0, ag0, s1 );
        _mm512_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

        s0 = _mm512_fmadd_pd( w2m, ah2, _mm512_mul_pd( w3m, ah3 ) );
        s1 = _mm512_fmadd_pd( w1m, ah1, s0 );
        w = _mm512_fmadd_pd( w0m, ah0, s1 );
        _mm512_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

        s0 = _mm512_fmadd_pd( w2m, ag2, _mm512_mul_pd( w3m, ag3 ) );
        s1 = _mm512_fmadd_pd( w1m, ag1, s0 );
        w = _mm512_fmadd_pd( w0m, ag0, s1 );
        _mm512_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );
    }

}

#endif // __AVX512F__

#ifdef __AVX2__

/* Based on the order of operations performend by fma2 */

void dda4mt2_fma2_reuse( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m256d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    for( j = 0 ; j < M / 2; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){

            a0 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + ( 2*i + 4 ) % N], A[(j*2)*lda + 2*i + 2 ], A[(j*2)*lda + 2*i] );
            a1 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + ( 2*i + 5 ) % N],  A[ (j*2)*lda + 2*i + 3 ],   A[ (j*2)*lda + 2*i + 1] );
            a2 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 8 ) % N],  A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + 2*i + 4 ],   A[ (j*2)*lda + 2*i + 2 ]);
            a3 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 9 ) % N],  A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + 2*i + 5 ],   A[ (j*2)*lda + 2*i + 3 ] );

            a4 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + ( 2*i + 4 ) % N], A[(j*2+1)*lda + 2*i + 2 ], A[(j*2+1)*lda + 2*i] );
            a5 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + ( 2*i + 5 ) % N],  A[ (j*2+1)*lda + 2*i + 3 ],   A[ (j*2+1)*lda + 2*i + 1] );
            a6 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 8 ) % N],  A[ (j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + 2*i + 4 ],   A[ (j*2+1)*lda + 2*i + 2 ]);
            a7 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 9 ) % N],  A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + 2*i + 5 ],   A[ (j*2+1)*lda + 2*i + 3 ] );

            a8 = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 4 ) % N], A[((j*2+2)%M)*lda + 2*i + 2 ], A[((j*2+2)%M)*lda + 2*i] );
            a9 = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 5 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 3 ],   A[ ((j*2+2)%M)*lda + 2*i + 1] );
            aa = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 4 ],   A[ ((j*2+2)%M)*lda + 2*i + 2 ]);
            ab = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 5 ],   A[ ((j*2+2)%M)*lda + 2*i + 3 ] );

            ac = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 4 ) % N], A[((j*2+3)%M)*lda + 2*i + 2 ], A[((j*2+3)%M)*lda + 2*i] );
            ad = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 5 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 3 ],   A[ ((j*2+3)%M)*lda + 2*i + 1] );
            ae = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 4 ],   A[ ((j*2+3)%M)*lda + 2*i + 2 ]);
            af = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 5 ],   A[ ((j*2+3)%M)*lda + 2*i + 3 ] );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
            s1 = _mm256_fmadd_pd( a1, ah1, s0 );
            w0 = _mm256_fmadd_pd( a0, ah0, s1 );

            s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
            s1 = _mm256_fmadd_pd( a1, ag1, s0 );
            w0m = _mm256_fmadd_pd( a0, ag0, s1 );

            s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
            s1 = _mm256_fmadd_pd( a5, ah1, s0 );
            w1 = _mm256_fmadd_pd( a4, ah0, s1 );

            s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
            s1 = _mm256_fmadd_pd( a5, ag1, s0 );
            w1m = _mm256_fmadd_pd( a4, ag0, s1 );

            s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
            s1 = _mm256_fmadd_pd( a9, ah1, s0 );
            w2 = _mm256_fmadd_pd( a8, ah0, s1 );

            s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
            s1 = _mm256_fmadd_pd( a9, ag1, s0 );
            w2m = _mm256_fmadd_pd( a8, ag0, s1 );

            s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
            s1 = _mm256_fmadd_pd( ad, ah1, s0 );
            w3 = _mm256_fmadd_pd( ac, ah0, s1 );

            s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
            s1 = _mm256_fmadd_pd( ad, ag1, s0 );
            w3m = _mm256_fmadd_pd( ac, ag0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
            s1 = _mm256_fmadd_pd( w1, ah1, s0 );
            w = _mm256_fmadd_pd( w0, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
            s1 = _mm256_fmadd_pd( w1, ag1, s0 );
            w = _mm256_fmadd_pd( w0, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
            s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
            w = _mm256_fmadd_pd( w0m, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

            s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
            s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
            w = _mm256_fmadd_pd( w0m, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );

        }
   }
}

void dda4mt2_fma2_reuse_gather( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m256d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256i stride = _mm256_set_epi64x( 3*sizeof( double ), 2*sizeof( double ),
                                          sizeof( double ), 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    for( j = 0 ; j < M / 2; j++ ) {
        for( i = 0 ; i < N / 2 - 4 ; i+=4 ){
            a0 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i], stride, 2 );
            a1 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i + 1], stride, 2 );
            a2 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i + 2], stride, 2 );
            a3 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i + 3], stride, 2 );

            a4 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i], stride, 2 );
            a5 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 1], stride, 2 );
            a6 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 2], stride, 2 );
            a7 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 3], stride, 2 );

            a8 = _mm256_i64gather_pd( &A[((j*2+2)%M)*lda + 2*i], stride, 2 );
            a9 = _mm256_i64gather_pd( &A[((j*2+2)%M)*lda + 2*i + 1], stride, 2 );
            aa = _mm256_i64gather_pd( &A[((j*2+2)%M)*lda + 2*i + 2], stride, 2 );
            ab = _mm256_i64gather_pd( &A[((j*2+2)%M)*lda + 2*i + 3], stride, 2 );

            ac = _mm256_i64gather_pd( &A[((j*2+3)%M)*lda + 2*i], stride, 2 );
            ad = _mm256_i64gather_pd( &A[((j*2+3)%M)*lda + 2*i + 1], stride, 2 );
            ae = _mm256_i64gather_pd( &A[((j*2+3)%M)*lda + 2*i + 2], stride, 2 );
            af = _mm256_i64gather_pd( &A[((j*2+3)%M)*lda + 2*i + 3], stride, 2 );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
            s1 = _mm256_fmadd_pd( a1, ah1, s0 );
            w0 = _mm256_fmadd_pd( a0, ah0, s1 );

            s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
            s1 = _mm256_fmadd_pd( a1, ag1, s0 );
            w0m = _mm256_fmadd_pd( a0, ag0, s1 );

            s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
            s1 = _mm256_fmadd_pd( a5, ah1, s0 );
            w1 = _mm256_fmadd_pd( a4, ah0, s1 );

            s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
            s1 = _mm256_fmadd_pd( a5, ag1, s0 );
            w1m = _mm256_fmadd_pd( a4, ag0, s1 );

            s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
            s1 = _mm256_fmadd_pd( a9, ah1, s0 );
            w2 = _mm256_fmadd_pd( a8, ah0, s1 );

            s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
            s1 = _mm256_fmadd_pd( a9, ag1, s0 );
            w2m = _mm256_fmadd_pd( a8, ag0, s1 );

            s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
            s1 = _mm256_fmadd_pd( ad, ah1, s0 );
            w3 = _mm256_fmadd_pd( ac, ah0, s1 );

            s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
            s1 = _mm256_fmadd_pd( ad, ag1, s0 );
            w3m = _mm256_fmadd_pd( ac, ag0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
            s1 = _mm256_fmadd_pd( w1, ah1, s0 );
            w = _mm256_fmadd_pd( w0, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
            s1 = _mm256_fmadd_pd( w1, ag1, s0 );
            w = _mm256_fmadd_pd( w0, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
            s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
            w = _mm256_fmadd_pd( w0m, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

            s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
            s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
            w = _mm256_fmadd_pd( w0m, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );

        }

        i = N/2 - 4;

        /* The last column cannot be done with gather because of the folding */

        a0 = _mm256_i64gather_pd( &A[2*j*lda + 2*i], stride, 2 );
        a1 = _mm256_i64gather_pd( &A[2*j*lda + 2*i + 1], stride, 2 );
        a2 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 8 ) % N],  A[ (j*2)*lda + ( 2*i + 6 ) % N],  A[ (j*2)*lda + 2*i + 4 ],   A[ (j*2)*lda + 2*i + 2 ]);
        a3 = _mm256_set_pd( A[ (j*2)*lda + ( 2*i + 9 ) % N],  A[ (j*2)*lda + ( 2*i + 7 ) % N],  A[ (j*2)*lda + 2*i + 5 ],   A[ (j*2)*lda + 2*i + 3 ] );

        a4 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i], stride, 2 );
        a5 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 1], stride, 2 );
        a6 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 8 ) % N],  A[ (j*2+1)*lda + ( 2*i + 6 ) % N],  A[ (j*2+1)*lda + 2*i + 4 ],   A[ (j*2+1)*lda + 2*i + 2 ]);
        a7 = _mm256_set_pd( A[ (j*2+1)*lda + ( 2*i + 9 ) % N],  A[ (j*2+1)*lda + ( 2*i + 7 ) % N],  A[ (j*2+1)*lda + 2*i + 5 ],   A[ (j*2+1)*lda + 2*i + 3 ] );

        a8 = _mm256_i64gather_pd( &A[((j*2+2)%M)*lda + 2*i], stride, 2 );
        a9 = _mm256_i64gather_pd( &A[((j*2+2)%M)*lda + 2*i + 1], stride, 2 );
        aa = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 4 ],   A[ ((j*2+2)%M)*lda + 2*i + 2 ]);
        ab = _mm256_set_pd( A[ ((j*2+2)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+2)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+2)%M)*lda + 2*i + 5 ],   A[ ((j*2+2)%M)*lda + 2*i + 3 ] );

        ac = _mm256_i64gather_pd( &A[((j*2+3)%M)*lda + 2*i], stride, 2 );
        ad = _mm256_i64gather_pd( &A[((j*2+3)%M)*lda + 2*i + 1], stride, 2 );
        ae = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 8 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 6 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 4 ],   A[ ((j*2+3)%M)*lda + 2*i + 2 ]);
        af = _mm256_set_pd( A[ ((j*2+3)%M)*lda + ( 2*i + 9 ) % N],  A[ ((j*2+3)%M)*lda + ( 2*i + 7 ) % N],  A[ ((j*2+3)%M)*lda + 2*i + 5 ],   A[ ((j*2+3)%M)*lda + 2*i + 3 ] );

        /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

        s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
        s1 = _mm256_fmadd_pd( a1, ah1, s0 );
        w0 = _mm256_fmadd_pd( a0, ah0, s1 );

        s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
        s1 = _mm256_fmadd_pd( a1, ag1, s0 );
        w0m = _mm256_fmadd_pd( a0, ag0, s1 );

        s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
        s1 = _mm256_fmadd_pd( a5, ah1, s0 );
        w1 = _mm256_fmadd_pd( a4, ah0, s1 );

        s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
        s1 = _mm256_fmadd_pd( a5, ag1, s0 );
        w1m = _mm256_fmadd_pd( a4, ag0, s1 );

        s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
        s1 = _mm256_fmadd_pd( a9, ah1, s0 );
        w2 = _mm256_fmadd_pd( a8, ah0, s1 );

        s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
        s1 = _mm256_fmadd_pd( a9, ag1, s0 );
        w2m = _mm256_fmadd_pd( a8, ag0, s1 );

        s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
        s1 = _mm256_fmadd_pd( ad, ah1, s0 );
        w3 = _mm256_fmadd_pd( ac, ah0, s1 );

        s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
        s1 = _mm256_fmadd_pd( ad, ag1, s0 );
        w3m = _mm256_fmadd_pd( ac, ag0, s1 );

        /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

        s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
        s1 = _mm256_fmadd_pd( w1, ah1, s0 );
        w = _mm256_fmadd_pd( w0, ah0, s1 );
        _mm256_storeu_pd( &B[ 1*j*ldb + i ], w );

        s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
        s1 = _mm256_fmadd_pd( w1, ag1, s0 );
        w = _mm256_fmadd_pd( w0, ag0, s1 );
        _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

        s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
        s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
        w = _mm256_fmadd_pd( w0m, ah0, s1 );
        _mm256_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

        s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
        s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
        w = _mm256_fmadd_pd( w0m, ag0, s1 );
        _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );
    }
}

void dda4mt2_fma2_reuse_gather_peel( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m256d w, s0, s1, w0, w0m, w1, w1m, w2, w2m, w3, w3m;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256i stride = _mm256_set_epi64x( 3*sizeof( double ), 2*sizeof( double ),
                                          sizeof( double ), 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* Main loop */
    
    for( j = 0 ; j < M / 2 - 1; j++ ) {
        for( i = 0 ; i < N / 2 - 4 ; i+=4 ){
            a0 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i], stride, 2 );
            a1 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i + 1], stride, 2 );
            a2 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i + 2], stride, 2 );
            a3 = _mm256_i64gather_pd( &A[(j*2)*lda + 2*i + 3], stride, 2 );

            a4 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i], stride, 2 );
            a5 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 1], stride, 2 );
            a6 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 2], stride, 2 );
            a7 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 3], stride, 2 );

            a8 = _mm256_i64gather_pd( &A[(j*2+2)*lda + 2*i], stride, 2 );
            a9 = _mm256_i64gather_pd( &A[(j*2+2)*lda + 2*i + 1], stride, 2 );
            aa = _mm256_i64gather_pd( &A[(j*2+2)*lda + 2*i + 2], stride, 2 );
            ab = _mm256_i64gather_pd( &A[(j*2+2)*lda + 2*i + 3], stride, 2 );

            ac = _mm256_i64gather_pd( &A[(j*2+3)*lda + 2*i], stride, 2 );
            ad = _mm256_i64gather_pd( &A[(j*2+3)*lda + 2*i + 1], stride, 2 );
            ae = _mm256_i64gather_pd( &A[(j*2+3)*lda + 2*i + 2], stride, 2 );
            af = _mm256_i64gather_pd( &A[(j*2+3)*lda + 2*i + 3], stride, 2 );

            /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

            s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
            s1 = _mm256_fmadd_pd( a1, ah1, s0 );
            w0 = _mm256_fmadd_pd( a0, ah0, s1 );

            s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
            s1 = _mm256_fmadd_pd( a1, ag1, s0 );
            w0m = _mm256_fmadd_pd( a0, ag0, s1 );

            s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
            s1 = _mm256_fmadd_pd( a5, ah1, s0 );
            w1 = _mm256_fmadd_pd( a4, ah0, s1 );

            s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
            s1 = _mm256_fmadd_pd( a5, ag1, s0 );
            w1m = _mm256_fmadd_pd( a4, ag0, s1 );

            s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
            s1 = _mm256_fmadd_pd( a9, ah1, s0 );
            w2 = _mm256_fmadd_pd( a8, ah0, s1 );

            s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
            s1 = _mm256_fmadd_pd( a9, ag1, s0 );
            w2m = _mm256_fmadd_pd( a8, ag0, s1 );

            s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
            s1 = _mm256_fmadd_pd( ad, ah1, s0 );
            w3 = _mm256_fmadd_pd( ac, ah0, s1 );

            s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
            s1 = _mm256_fmadd_pd( ad, ag1, s0 );
            w3m = _mm256_fmadd_pd( ac, ag0, s1 );

            /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

            s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
            s1 = _mm256_fmadd_pd( w1, ah1, s0 );
            w = _mm256_fmadd_pd( w0, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
            s1 = _mm256_fmadd_pd( w1, ag1, s0 );
            w = _mm256_fmadd_pd( w0, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

            s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
            s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
            w = _mm256_fmadd_pd( w0m, ah0, s1 );
            _mm256_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

            s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
            s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
            w = _mm256_fmadd_pd( w0m, ag0, s1 );
            _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );

        }

        i = N/2 - 4;

        /* The last column cannot be done with gather because of the folding */

        a0 = _mm256_i64gather_pd( &A[2*j*lda + 2*i], stride, 2 );
        a1 = _mm256_i64gather_pd( &A[2*j*lda + 2*i + 1], stride, 2 );
        a2 = _mm256_set_pd( A[ j*2*lda ],  A[ j*2*lda + 2*i + 6 ],  A[ j*2*lda + 2*i + 4 ],   A[ j*2*lda + 2*i + 2 ]);
        a3 = _mm256_set_pd( A[ j*2*lda + 1 ],  A[ j*2*lda + 2*i + 7 ],  A[ j*2*lda + 2*i + 5 ],   A[ j*2*lda + 2*i + 3 ] );

        a4 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i], stride, 2 );
        a5 = _mm256_i64gather_pd( &A[(j*2+1)*lda + 2*i + 1], stride, 2 );
        a6 = _mm256_set_pd( A[ (j*2+1)*lda ],  A[ (j*2+1)*lda + 2*i + 6 ],  A[ (j*2+1)*lda + 2*i + 4 ],   A[ (j*2+1)*lda + 2*i + 2 ]);
        a7 = _mm256_set_pd( A[ (j*2+1)*lda + 1],  A[ (j*2+1)*lda + 2*i + 7],  A[ (j*2+1)*lda + 2*i + 5 ],   A[ (j*2+1)*lda + 2*i + 3 ] );

        a8 = _mm256_i64gather_pd( &A[(j*2+2)*lda + 2*i], stride, 2 );
        a9 = _mm256_i64gather_pd( &A[(j*2+2)*lda + 2*i + 1], stride, 2 );
        aa = _mm256_set_pd( A[ (j*2+2)*lda ],  A[ (j*2+2)*lda + 2*i + 6 ],  A[ (j*2+2)*lda + 2*i + 4 ],   A[ (j*2+2)*lda + 2*i + 2 ]);
        ab = _mm256_set_pd( A[ (j*2+2)*lda + 1],  A[ (j*2+2)*lda + 2*i + 7 ],  A[ (j*2+2)*lda + 2*i + 5 ],   A[ (j*2+2)*lda + 2*i + 3 ] );

        ac = _mm256_i64gather_pd( &A[(j*2+3)*lda + 2*i], stride, 2 );
        ad = _mm256_i64gather_pd( &A[(j*2+3)*lda + 2*i + 1], stride, 2 );
        ae = _mm256_set_pd( A[ (j*2+3)*lda ],  A[ (j*2+3)*lda + 2*i + 6 ],  A[ (j*2+3)*lda + 2*i + 4 ],   A[ (j*2+3)*lda + 2*i + 2 ]);
        af = _mm256_set_pd( A[ (j*2+3)*lda + 1],  A[ (j*2+3)*lda + 2*i + 7 ],  A[ (j*2+3)*lda + 2*i + 5 ],   A[ (j*2+3)*lda + 2*i + 3 ] );

        /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */

        s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
        s1 = _mm256_fmadd_pd( a1, ah1, s0 );
        w0 = _mm256_fmadd_pd( a0, ah0, s1 );

        s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
        s1 = _mm256_fmadd_pd( a1, ag1, s0 );
        w0m = _mm256_fmadd_pd( a0, ag0, s1 );

        s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
        s1 = _mm256_fmadd_pd( a5, ah1, s0 );
        w1 = _mm256_fmadd_pd( a4, ah0, s1 );

        s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
        s1 = _mm256_fmadd_pd( a5, ag1, s0 );
        w1m = _mm256_fmadd_pd( a4, ag0, s1 );

        s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
        s1 = _mm256_fmadd_pd( a9, ah1, s0 );
        w2 = _mm256_fmadd_pd( a8, ah0, s1 );

        s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
        s1 = _mm256_fmadd_pd( a9, ag1, s0 );
        w2m = _mm256_fmadd_pd( a8, ag0, s1 );

        s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
        s1 = _mm256_fmadd_pd( ad, ah1, s0 );
        w3 = _mm256_fmadd_pd( ac, ah0, s1 );

        s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
        s1 = _mm256_fmadd_pd( ad, ag1, s0 );
        w3m = _mm256_fmadd_pd( ac, ag0, s1 );

        /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */

        s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
        s1 = _mm256_fmadd_pd( w1, ah1, s0 );
        w = _mm256_fmadd_pd( w0, ah0, s1 );
        _mm256_storeu_pd( &B[ 1*j*ldb + i ], w );

        s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
        s1 = _mm256_fmadd_pd( w1, ag1, s0 );
        w = _mm256_fmadd_pd( w0, ag0, s1 );
        _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i ], w );

        s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
        s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
        w = _mm256_fmadd_pd( w0m, ah0, s1 );
        _mm256_storeu_pd( &B[ 1*j*ldb + i + N/2 ], w );

        s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
        s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
        w = _mm256_fmadd_pd( w0m, ag0, s1 );
        _mm256_storeu_pd( &B[ (1*j + M/2)*ldb + i + N / 2 ], w );
    }

    /* Peeling */

    for( i = 0 ; i < N / 2 - 4 ; i+=4 ){
      a0 = _mm256_i64gather_pd( &A[ (M - 2)*lda + 2*i], stride, 2 );
      a1 = _mm256_i64gather_pd( &A[ (M - 2)*lda + 2*i + 1], stride, 2 );
      a2 = _mm256_i64gather_pd( &A[ (M - 2)*lda + 2*i + 2], stride, 2 );
      a3 = _mm256_i64gather_pd( &A[ (M - 2)*lda + 2*i + 3], stride, 2 );
      
      a4 = _mm256_i64gather_pd( &A[ (M - 1)*lda + 2*i], stride, 2 );
      a5 = _mm256_i64gather_pd( &A[ (M - 1)*lda + 2*i + 1], stride, 2 );
      a6 = _mm256_i64gather_pd( &A[ (M - 1)*lda + 2*i + 2], stride, 2 );
      a7 = _mm256_i64gather_pd( &A[ (M - 1)*lda + 2*i + 3], stride, 2 );
      
      a8 = _mm256_i64gather_pd( &A[ 2*i], stride, 2 );
      a9 = _mm256_i64gather_pd( &A[ 2*i + 1], stride, 2 );
      aa = _mm256_i64gather_pd( &A[ 2*i + 2], stride, 2 );
      ab = _mm256_i64gather_pd( &A[ 2*i + 3], stride, 2 );
      
      ac = _mm256_i64gather_pd( &A[ lda + 2*i], stride, 2 );
      ad = _mm256_i64gather_pd( &A[ lda + 2*i + 1], stride, 2 );
      ae = _mm256_i64gather_pd( &A[ lda + 2*i + 2], stride, 2 );
      af = _mm256_i64gather_pd( &A[ lda + 2*i + 3], stride, 2 );

      /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */
      
      s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
      s1 = _mm256_fmadd_pd( a1, ah1, s0 );
      w0 = _mm256_fmadd_pd( a0, ah0, s1 );
      
      s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
      s1 = _mm256_fmadd_pd( a1, ag1, s0 );
      w0m = _mm256_fmadd_pd( a0, ag0, s1 );
      
      s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
      s1 = _mm256_fmadd_pd( a5, ah1, s0 );
      w1 = _mm256_fmadd_pd( a4, ah0, s1 );
      
      s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
      s1 = _mm256_fmadd_pd( a5, ag1, s0 );
      w1m = _mm256_fmadd_pd( a4, ag0, s1 );
      
      s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
      s1 = _mm256_fmadd_pd( a9, ah1, s0 );
      w2 = _mm256_fmadd_pd( a8, ah0, s1 );
      
      s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
      s1 = _mm256_fmadd_pd( a9, ag1, s0 );
      w2m = _mm256_fmadd_pd( a8, ag0, s1 );
      
      s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
      s1 = _mm256_fmadd_pd( ad, ah1, s0 );
      w3 = _mm256_fmadd_pd( ac, ah0, s1 );
      
      s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
      s1 = _mm256_fmadd_pd( ad, ag1, s0 );
      w3m = _mm256_fmadd_pd( ac, ag0, s1 );
      
      /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */
      
      s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
      s1 = _mm256_fmadd_pd( w1, ah1, s0 );
      w = _mm256_fmadd_pd( w0, ah0, s1 );
      _mm256_storeu_pd( &B[ ( M / 2 - 1 )*ldb + i ], w );
      
      s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
      s1 = _mm256_fmadd_pd( w1, ag1, s0 );
      w = _mm256_fmadd_pd( w0, ag0, s1 );
      _mm256_storeu_pd( &B[ ( M - 1 )*ldb + i ], w );
      
      s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
      s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
      w = _mm256_fmadd_pd( w0m, ah0, s1 );
      _mm256_storeu_pd( &B[ ( M / 2 - 1 )*ldb + i + N/2 ], w );

      s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
      s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
      w = _mm256_fmadd_pd( w0m, ag0, s1 );
      _mm256_storeu_pd( &B[ ( M - 1 )*ldb + i + N / 2 ], w );
      
    }
    
    i = N/2 - 4;

    /* The last column cannot be done with gather because of the folding */
    
    a0 = _mm256_i64gather_pd( &A[ 2*(M/2 - 1)*lda + 2*i], stride, 2 );
    a1 = _mm256_i64gather_pd( &A[ 2*(M/2 - 1)*lda + 2*i + 1], stride, 2 );
    a2 = _mm256_set_pd( A[ (M - 2)*lda ],  A[ (M - 2)*lda + 2*i + 6 ],  A[ (M - 2)*lda + 2*i + 4 ],   A[ (M - 2)*lda + 2*i + 2 ]);
    a3 = _mm256_set_pd( A[ (M - 2)*lda + 1 ],  A[ (M - 2)*lda + 2*i + 7 ],  A[ (M - 2)*lda + 2*i + 5 ],   A[ ( M - 2)*lda + 2*i + 3 ] );

    a4 = _mm256_i64gather_pd( &A[(M - 1)*lda + 2*i], stride, 2 );
    a5 = _mm256_i64gather_pd( &A[(M - 1)*lda + 2*i + 1], stride, 2 );
    a6 = _mm256_set_pd( A[ (M - 1)*lda ],  A[ (M - 1)*lda + 2*i + 6 ],  A[ (M - 1)*lda + 2*i + 4 ],   A[ (M - 1)*lda + 2*i + 2 ]);
    a7 = _mm256_set_pd( A[ (M - 1)*lda + 1 ],  A[ (M - 1)*lda + 2*i + 7 ],  A[ (M - 1)*lda + 2*i + 5 ],   A[ (M - 1)*lda + 2*i + 3 ] );

    a8 = _mm256_i64gather_pd( &A[ 2*i], stride, 2 );
    a9 = _mm256_i64gather_pd( &A[2*i + 1], stride, 2 );
    aa = _mm256_set_pd( A[ 0 ],  A[ 2*i + 6 ],  A[ 2*i + 4 ],   A[ 2*i + 2 ]);
    ab = _mm256_set_pd( A[ 1 ],  A[ 2*i + 7 ],  A[ 2*i + 5 ],   A[ 2*i + 3 ] );

    ac = _mm256_i64gather_pd( &A[ lda + 2*i], stride, 2 );
    ad = _mm256_i64gather_pd( &A[ lda + 2*i + 1], stride, 2 );
    ae = _mm256_set_pd( A[ lda ],  A[ lda + 2*i + 6 ],  A[ 2*i + 4 ],   A[ 2*i + 2 ]);
    af = _mm256_set_pd( A[ lda + 1 ],  A[ lda + 2*i + 7 ],  A[ lda + 2*i + 5 ],   A[ lda + 2*i + 3 ] );
    
    /* Lines: W1 = A[i][j] + A[i+1][j] + A[i+2][j] + A[i+3][j] = A1 + A2 + A3 + A4 */
    
    s0 = _mm256_fmadd_pd( a2, ah2, _mm256_mul_pd( a3, ah3 ) );
    s1 = _mm256_fmadd_pd( a1, ah1, s0 );
    w0 = _mm256_fmadd_pd( a0, ah0, s1 );
    
    s0 = _mm256_fmadd_pd( a2, ag2, _mm256_mul_pd( a3, ag3 ) );
    s1 = _mm256_fmadd_pd( a1, ag1, s0 );
    w0m = _mm256_fmadd_pd( a0, ag0, s1 );
    
    s0 = _mm256_fmadd_pd( a6, ah2, _mm256_mul_pd( a7, ah3 ) );
    s1 = _mm256_fmadd_pd( a5, ah1, s0 );
    w1 = _mm256_fmadd_pd( a4, ah0, s1 );
    
    s0 = _mm256_fmadd_pd( a6, ag2, _mm256_mul_pd( a7, ag3 ) );
    s1 = _mm256_fmadd_pd( a5, ag1, s0 );
    w1m = _mm256_fmadd_pd( a4, ag0, s1 );
    
    s0 = _mm256_fmadd_pd( aa, ah2, _mm256_mul_pd( ab, ah3 ) );
    s1 = _mm256_fmadd_pd( a9, ah1, s0 );
    w2 = _mm256_fmadd_pd( a8, ah0, s1 );
    
    s0 = _mm256_fmadd_pd( aa, ag2, _mm256_mul_pd( ab, ag3 ) );
    s1 = _mm256_fmadd_pd( a9, ag1, s0 );
    w2m = _mm256_fmadd_pd( a8, ag0, s1 );
    
    s0 = _mm256_fmadd_pd( ae, ah2, _mm256_mul_pd( af, ah3 ) );
    s1 = _mm256_fmadd_pd( ad, ah1, s0 );
    w3 = _mm256_fmadd_pd( ac, ah0, s1 );
    
    s0 = _mm256_fmadd_pd( ae, ag2, _mm256_mul_pd( af, ag3 ) );
    s1 = _mm256_fmadd_pd( ad, ag1, s0 );
    w3m = _mm256_fmadd_pd( ac, ag0, s1 );
    
    /* Columns: B = W1 + W2 + W3 + W4 = W[i][j] + W[j][j+1] + W[j][j+2] + W[j][j+3] */
    
    s0 = _mm256_fmadd_pd( w2, ah2, _mm256_mul_pd( w3, ah3 ) );
    s1 = _mm256_fmadd_pd( w1, ah1, s0 );
    w = _mm256_fmadd_pd( w0, ah0, s1 );
    _mm256_storeu_pd( &B[ ( M / 2 - 1 )*ldb + i ], w );
    
    s0 = _mm256_fmadd_pd( w2, ag2, _mm256_mul_pd( w3, ag3 ) );
    s1 = _mm256_fmadd_pd( w1, ag1, s0 );
    w = _mm256_fmadd_pd( w0, ag0, s1 );
    _mm256_storeu_pd( &B[ ( M - 1 )*ldb + i ], w );
    
    s0 = _mm256_fmadd_pd( w2m, ah2, _mm256_mul_pd( w3m, ah3 ) );
    s1 = _mm256_fmadd_pd( w1m, ah1, s0 );
    w = _mm256_fmadd_pd( w0m, ah0, s1 );
    _mm256_storeu_pd( &B[ ( M / 2 - 1 )*ldb + i + N/2 ], w );
    
    s0 = _mm256_fmadd_pd( w2m, ag2, _mm256_mul_pd( w3m, ag3 ) );
    s1 = _mm256_fmadd_pd( w1m, ag1, s0 );
    w = _mm256_fmadd_pd( w0m, ag0, s1 );
    _mm256_storeu_pd( &B[ (M - 1)*ldb + i + N / 2 ], w );
    
}

#endif // __AVX2__

/*
c     Compute 2D Daubechies D4 inverse transform of a matrix
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
c     di4 Daubechies D4 inverse
c     mt matrix transform
c     2 2D
*/

#ifdef  __cplusplus
void ddi4mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int j, k;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;

    /* dim 1 */

    for( k = 0 ; k < M ; k++ ) {
        for( j = 0 ; j < N / 2 ; j++ ) {
            W[ k*ldb + 2*j] = h0 * A[k*N + j] + g0 * A[ k*N + N / 2 + j]
                + h2 * A[k*N + ( j - 1 + N/2 ) % (N/2)] + g2 * A[ k*N + N / 2 + ( ( j - 1 + N/2) %(  N/2))];
            W[ k*ldb + 2*j + 1] = h1 * A[k*N + j] + g1 * A[ k*N + N / 2 + j]
                + h3 * A[k*N + ( j - 1 + N/2 ) % (N/2) ] + g3 * A[ k*N + N / 2 + ( j - 1 + N/2) % ( N / 2 )];
        }
    }

    /* dim 2 */

    for( k = 0 ; k < M / 2 ; k++ ) {
        for( j = 0 ; j < N ; j++ ) {
            B[ 2*k*N + j] = h0 * W[k*lda + j] + g0 * W[ (k + M/2)*lda + j]
                + h2 * W[((k-1+M/2)%(M/2))*lda + j] + g2 * W[ (M/2 + (k-1+M/2)%(M/2))*lda + j];
            B[ (2*k+1)*N + j] =  h1 * W[k*lda + j] + g1 * W[ (k + M/2)*N + j]
            + h3 * W[((k-1+M/2)%(M/2))*lda + j] + g3 * W[ (( k-1+M/2)%(M/2)+M/2)*lda + j];
        }
    }

}

#if defined( __SSE__ ) || defined( __aarch64__ )

void ddi4mt2_sse( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m128d w, w0, w1, w2, w3, s0, s1;
     __m128d a0, a1, a2, a3;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;
    __m128d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm_set_pd( h1, h0);
    gbegin = _mm_set_pd( g1, g0 );
    hend = _mm_set_pd( h3, h2);
    gend = _mm_set_pd( g3, g2 );
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
      for( i = 0 ; i < N / 2 ; i++ ) {
            a0 = _mm_set1_pd( A[ j*N + i] );
            a1 = _mm_set1_pd( A[ j*N + N / 2 + i] );
            a2 = _mm_set1_pd( A[ j*N + ( i - 1 + N/2 ) % (N/2) ] );
            a3 = _mm_set1_pd( A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            w0 = _mm_mul_pd( a0, hbegin );
            w1 = _mm_mul_pd( a1, gbegin );
            w2 = _mm_mul_pd( a2, hend );
            w3 = _mm_mul_pd( a3, gend );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=2 ) {
            a0 = _mm_loadu_pd( &W[ j*N + i] );
            a1 = _mm_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            w0 = _mm_mul_pd( a0, ah0 );
            w1 = _mm_mul_pd( a1, ag0 );
            w2 = _mm_mul_pd( a2, ah2 );
            w3 = _mm_mul_pd( a3, ag2 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ 2*j*ldb + i], w );

            w0 = _mm_mul_pd( a0, ah1 );
            w1 = _mm_mul_pd( a1, ag1 );
            w2 = _mm_mul_pd( a2, ah3 );
            w3 = _mm_mul_pd( a3, ag3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ (2*j+1)*ldb + i], w );

        }
    }
}

void ddi4mt2_sse_peel( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m128d w, w0, w1, w2, w3, s0, s1;
     __m128d a0, a1, a2, a3;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;
    __m128d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm_set_pd( h1, h0);
    gbegin = _mm_set_pd( g1, g0 );
    hend = _mm_set_pd( h3, h2);
    gend = _mm_set_pd( g3, g2 );
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
      
      /* Peeling */
      
      a0 = _mm_set1_pd( A[ j*N] );
      a1 = _mm_set1_pd( A[ j*N + N / 2] );
      a2 = _mm_set1_pd( A[ j*N + N/2 - 1 ] );
      a3 = _mm_set1_pd( A[ j*N + N - 1 ] );
      
      w0 = _mm_mul_pd( a0, hbegin );
      w1 = _mm_mul_pd( a1, gbegin );
      w2 = _mm_mul_pd( a2, hend );
      w3 = _mm_mul_pd( a3, gend );
      
      s0 = _mm_add_pd( w0, w1);
      s1 = _mm_add_pd( w2, w3);
      w = _mm_add_pd( s0, s1 );
      
      _mm_storeu_pd( &W[ j*N ], w );

      for( i = 1 ; i < N / 2 ; i++ ) {
            a0 = _mm_set1_pd( A[ j*N + i] );
            a1 = _mm_set1_pd( A[ j*N + N / 2 + i] );
            a2 = _mm_set1_pd( A[ j*N + i - 1 ] );
            a3 = _mm_set1_pd( A[ j*N + N / 2 + i - 1] );

            w0 = _mm_mul_pd( a0, hbegin );
            w1 = _mm_mul_pd( a1, gbegin );
            w2 = _mm_mul_pd( a2, hend );
            w3 = _mm_mul_pd( a3, gend );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* dim 2 */

    /* Peeling */

    for( i = 0 ; i < N ; i+=2 ) {
      a0 = _mm_loadu_pd( &W[ i ] );
      a1 = _mm_loadu_pd( &W[ ( M/2 )*N + i] );
      a2 = _mm_loadu_pd( &W[ ( M/2 - 1 )*N + i] );
      a3 = _mm_loadu_pd( &W[ ( M-1 )*N + i] );
      
      w0 = _mm_mul_pd( a0, ah0 );
      w1 = _mm_mul_pd( a1, ag0 );
      w2 = _mm_mul_pd( a2, ah2 );
      w3 = _mm_mul_pd( a3, ag2 );
      
      s0 = _mm_add_pd( w0, w1);
      s1 = _mm_add_pd( w2, w3);
      w = _mm_add_pd( s0, s1 );
      
      _mm_storeu_pd( &B[ i ], w );
      
      w0 = _mm_mul_pd( a0, ah1 );
      w1 = _mm_mul_pd( a1, ag1 );
      w2 = _mm_mul_pd( a2, ah3 );
      w3 = _mm_mul_pd( a3, ag3 );
      
      s0 = _mm_add_pd( w0, w1);
      s1 = _mm_add_pd( w2, w3);
      w = _mm_add_pd( s0, s1 );
      
      _mm_storeu_pd( &B[ ldb + i], w );
    }

    for( j = 1 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=2 ) {
            a0 = _mm_loadu_pd( &W[ j*N + i] );
            a1 = _mm_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm_loadu_pd( &W[ (j-1)*N + i] );
            a3 = _mm_loadu_pd( &W[ (j-1+M/2)*N + i] );

            w0 = _mm_mul_pd( a0, ah0 );
            w1 = _mm_mul_pd( a1, ag0 );
            w2 = _mm_mul_pd( a2, ah2 );
            w3 = _mm_mul_pd( a3, ag2 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ 2*j*ldb + i], w );

            w0 = _mm_mul_pd( a0, ah1 );
            w1 = _mm_mul_pd( a1, ag1 );
            w2 = _mm_mul_pd( a2, ah3 );
            w3 = _mm_mul_pd( a3, ag3 );

            s0 = _mm_add_pd( w0, w1);
            s1 = _mm_add_pd( w2, w3);
            w = _mm_add_pd( s0, s1 );

            _mm_storeu_pd( &B[ (2*j+1)*ldb + i], w );

        }
    }

}

void ddi4mt2_sse_reuse( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m128d w0, w1, w2, w3, s0, s1, s2;
    __m128d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;
    __m128d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm_set_pd( h1, h0);
    gbegin = _mm_set_pd( g1, g0 );
    hend = _mm_set_pd( h3, h2);
    gend = _mm_set_pd( g3, g2 );
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );


    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N/2 ; i++ ) {
            a0 = _mm_set1_pd( A[ j*lda + i] );
            a1 = _mm_set1_pd( A[ j*lda + N / 2 + i] );
            a2 = _mm_set1_pd( A[ j*lda + ( i - 1 + N/2 ) % (N/2) ] );
            a3 = _mm_set1_pd( A[ j*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            a4 = _mm_set1_pd( A[ (j+M/2)*lda + i] );
            a5 = _mm_set1_pd( A[ (j+M/2)*lda + N / 2 + i] );
            a6 = _mm_set1_pd( A[ (j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
            a7 = _mm_set1_pd( A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            a8 = _mm_set1_pd( A[ ((j-1+M/2)%(M/2))*lda + i] );
            a9 = _mm_set1_pd( A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i] );
            aa = _mm_set1_pd( A[ ((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ] );
            ab = _mm_set1_pd( A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            ac = _mm_set1_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + i] );
            ad = _mm_set1_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i] );
            ae = _mm_set1_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
            af = _mm_set1_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */

            /* W[ j*N + 2*i] */
	    w0 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a0, hbegin ), _mm_mul_pd( a1, gbegin ) ),
			     _mm_add_pd( _mm_mul_pd( a2, hend ), _mm_mul_pd( a3, gend ) ) );

            /* W[ (j + M/2)*N + i] */
	    w1 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a4, hbegin ), _mm_mul_pd( a5, gbegin ) ),
			     _mm_add_pd( _mm_mul_pd( a6, hend ), _mm_mul_pd( a7, gend ) ) );

            /* W[((j-1+M/2)%(M/2))*N + i] */
	    w2 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a8, hbegin ), _mm_mul_pd( a9, gbegin ) ),
			     _mm_add_pd( _mm_mul_pd( aa, hend ), _mm_mul_pd( ab, gend ) ) );

            /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
	    w3 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( ac, hbegin ), _mm_mul_pd( ad, gbegin ) ),
			     _mm_add_pd( _mm_mul_pd( ae, hend ), _mm_mul_pd( af, gend ) ) );


	    s1 = _mm_add_pd( _mm_mul_pd( w2, ah2 ),  _mm_mul_pd( w3, ag2 ) );
            s0 = _mm_add_pd( _mm_mul_pd( w1, ag0 ), s1 );
            s2 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), s0 );

            _mm_storeu_pd( &B[ 2*j*ldb + 2*i], s2 );

	    s1 = _mm_add_pd( _mm_mul_pd( w2, ah3 ),  _mm_mul_pd( w3, ag3 ) );
            s0 = _mm_add_pd( _mm_mul_pd( w1, ag1 ),  s1 );
            s2 = _mm_add_pd( _mm_mul_pd( w0, ah1 ), s0 );

            _mm_storeu_pd( &B[ (2*j+1)*ldb + 2*i], s2 );

        }
    }
}

void ddi4mt2_sse_reuse_peel( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m128d w0, w1, w2, w3, s0, s1, s2;
    __m128d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;
    __m128d ah0, ah1, ah2, ah3;
    __m128d ag0, ag1, ag2, ag3;
    __m128d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm_set_pd( h1, h0);
    gbegin = _mm_set_pd( g1, g0 );
    hend = _mm_set_pd( h3, h2);
    gend = _mm_set_pd( g3, g2 );
    ah0 = _mm_set1_pd( h0 );
    ah1 = _mm_set1_pd( h1 );
    ah2 = _mm_set1_pd( h2 );
    ah3 = _mm_set1_pd( h3 );
    ag0 = _mm_set1_pd( g0 );
    ag1 = _mm_set1_pd( g1 );
    ag2 = _mm_set1_pd( g2 );
    ag3 = _mm_set1_pd( g3 );

    /* Peeling: top right corner */

    a0 = _mm_set1_pd( A[ 0 ] );
    a1 = _mm_set1_pd( A[ N / 2] );
    a2 = _mm_set1_pd( A[ N/2 - 1 ] );
    a3 = _mm_set1_pd( A[ N -1 ] );

    a4 = _mm_set1_pd( A[ (M/2)*lda ] );
    a5 = _mm_set1_pd( A[ (M/2)*lda + N / 2 ] );
    a6 = _mm_set1_pd( A[ (M/2)*lda + N / 2 - 1 ] );
    a7 = _mm_set1_pd( A[ (M/2)*lda + N - 1 ] );
    
    a8 = _mm_set1_pd( A[ ( M/2 - 1 )*lda ] );
    a9 = _mm_set1_pd( A[ ( M/2 - 1 )*lda + N / 2 ] );
    aa = _mm_set1_pd( A[ ( M/2 - 1 )*lda + N / 2 - 1 ] );
    ab = _mm_set1_pd( A[ ( M/2 - 1 )*lda + N - 1 ] );

    ac = _mm_set1_pd( A[ ( M - 1 )*lda ] );
    ad = _mm_set1_pd( A[ ( M - 1 )*lda + N / 2 ] );
    ae = _mm_set1_pd( A[ ( M - 1 )*lda + N / 2 - 1 ] );
    af = _mm_set1_pd( A[ ( M - 1 )*lda + N - 1 ] );
    
    /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
    
    /* W[ j*N + 2*i] */
    w0 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a0, hbegin ), _mm_mul_pd( a1, gbegin ) ),
		     _mm_add_pd( _mm_mul_pd( a2, hend ), _mm_mul_pd( a3, gend ) ) );
    
    /* W[ (j + M/2)*N + i] */
    w1 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a4, hbegin ), _mm_mul_pd( a5, gbegin ) ),
		     _mm_add_pd( _mm_mul_pd( a6, hend ), _mm_mul_pd( a7, gend ) ) );
    
    /* W[((j-1+M/2)%(M/2))*N + i] */
    w2 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a8, hbegin ), _mm_mul_pd( a9, gbegin ) ),
		     _mm_add_pd( _mm_mul_pd( aa, hend ), _mm_mul_pd( ab, gend ) ) );

    /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
    w3 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( ac, hbegin ), _mm_mul_pd( ad, gbegin ) ),
		     _mm_add_pd( _mm_mul_pd( ae, hend ), _mm_mul_pd( af, gend ) ) );
    
    
    s1 = _mm_add_pd( _mm_mul_pd( w2, ah2 ),  _mm_mul_pd( w3, ag2 ) );
    s0 = _mm_add_pd( _mm_mul_pd( w1, ag0 ), s1 );
    s2 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), s0 );

    _mm_storeu_pd( &B[ 0 ], s2 );
    
    s1 = _mm_add_pd( _mm_mul_pd( w2, ah3 ),  _mm_mul_pd( w3, ag3 ) );
    s0 = _mm_add_pd( _mm_mul_pd( w1, ag1 ),  s1 );
    s2 = _mm_add_pd( _mm_mul_pd( w0, ah1 ), s0 );
    
    _mm_storeu_pd( &B[ ldb ], s2 );
    
    /* Peeling: first lines */
    
    for( i = 1 ; i < N/2 ; i++ ) {
      a0 = _mm_set1_pd( A[ i] );
      a1 = _mm_set1_pd( A[ N / 2 + i ] );
      a2 = _mm_set1_pd( A[ i - 1 ] );
      a3 = _mm_set1_pd( A[ N / 2 + i - 1 ] );
      
      a4 = _mm_set1_pd( A[ (M/2)*lda + i] );
      a5 = _mm_set1_pd( A[ (M/2)*lda + N / 2 + i] );
      a6 = _mm_set1_pd( A[ (M/2)*lda + i - 1 ] );
      a7 = _mm_set1_pd( A[ (M/2)*lda + N / 2 + i - 1 ] );
	
      a8 = _mm_set1_pd( A[ ( M/2 - 1 )*lda + i] );
      a9 = _mm_set1_pd( A[ ( M/2 - 1 )*lda + N / 2 + i] );
      aa = _mm_set1_pd( A[ ( M/2 - 1 )*lda + i - 1 ] );
      ab = _mm_set1_pd( A[ ( M/2 - 1 )*lda + N / 2 + i - 1 ] );

      ac = _mm_set1_pd( A[ ( M - 1 )*lda + i] );
      ad = _mm_set1_pd( A[ ( M - 1 )*lda + N / 2 + i] );
      ae = _mm_set1_pd( A[ ( M - 1 )*lda + i - 1 ] );
      af = _mm_set1_pd( A[ ( M - 1 )*lda + N / 2 + i - 1 ] );
      
      /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
      
      /* W[ j*N + 2*i] */
      w0 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a0, hbegin ), _mm_mul_pd( a1, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( a2, hend ), _mm_mul_pd( a3, gend ) ) );
      
      /* W[ (j + M/2)*N + i] */
      w1 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a4, hbegin ), _mm_mul_pd( a5, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( a6, hend ), _mm_mul_pd( a7, gend ) ) );
      
      /* W[((j-1+M/2)%(M/2))*N + i] */
      w2 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a8, hbegin ), _mm_mul_pd( a9, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( aa, hend ), _mm_mul_pd( ab, gend ) ) );

      /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
      w3 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( ac, hbegin ), _mm_mul_pd( ad, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( ae, hend ), _mm_mul_pd( af, gend ) ) );
      
      
      s1 = _mm_add_pd( _mm_mul_pd( w2, ah2 ),  _mm_mul_pd( w3, ag2 ) );
      s0 = _mm_add_pd( _mm_mul_pd( w1, ag0 ), s1 );
      s2 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), s0 );
      
      _mm_storeu_pd( &B[ 2*i ], s2 );
      
      s1 = _mm_add_pd( _mm_mul_pd( w2, ah3 ),  _mm_mul_pd( w3, ag3 ) );
      s0 = _mm_add_pd( _mm_mul_pd( w1, ag1 ),  s1 );
      s2 = _mm_add_pd( _mm_mul_pd( w0, ah1 ), s0 );
      
      _mm_storeu_pd( &B[ ldb + 2*i], s2 );
    }
    
    for( j = 1 ; j < M/2 ; j++ ) {
      
      /* Peeling: first column */
      
      a0 = _mm_set1_pd( A[ j*lda ] );
      a1 = _mm_set1_pd( A[ j*lda + N / 2 ] );
      a2 = _mm_set1_pd( A[ j*lda + N / 2 - 1 ] );
      a3 = _mm_set1_pd( A[ j*lda + N - 1 ] );
      
      a4 = _mm_set1_pd( A[ (j+M/2)*lda ] );
      a5 = _mm_set1_pd( A[ (j+M/2)*lda + N / 2 ] );
      a6 = _mm_set1_pd( A[ (j+M/2)*lda + N / 2 - 1 ] );
      a7 = _mm_set1_pd( A[ (j+M/2)*lda + N - 1 ] );
      
      a8 = _mm_set1_pd( A[ ( j-1 )*lda ] );
      a9 = _mm_set1_pd( A[ ( j-1 )*lda + N / 2 ] );
      aa = _mm_set1_pd( A[ ( j-1 )*lda + N / 2 - 1 ] );
      ab = _mm_set1_pd( A[ ( j-1 )*lda + N - 1 ] );
      
      ac = _mm_set1_pd( A[ ( j-1 + M/2 )*lda ] );
      ad = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + N / 2 ] );
      ae = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + N / 2 - 1 ] );
      af = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + N - 1 ] );

      /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
      
      /* W[ j*N + 2*i] */
      w0 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a0, hbegin ), _mm_mul_pd( a1, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( a2, hend ), _mm_mul_pd( a3, gend ) ) );
      
      /* W[ (j + M/2)*N + i] */
      w1 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a4, hbegin ), _mm_mul_pd( a5, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( a6, hend ), _mm_mul_pd( a7, gend ) ) );
      
      /* W[((j-1+M/2)%(M/2))*N + i] */
      w2 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a8, hbegin ), _mm_mul_pd( a9, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( aa, hend ), _mm_mul_pd( ab, gend ) ) );
      
      /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
      w3 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( ac, hbegin ), _mm_mul_pd( ad, gbegin ) ),
		       _mm_add_pd( _mm_mul_pd( ae, hend ), _mm_mul_pd( af, gend ) ) );
      
      
      s1 = _mm_add_pd( _mm_mul_pd( w2, ah2 ),  _mm_mul_pd( w3, ag2 ) );
      s0 = _mm_add_pd( _mm_mul_pd( w1, ag0 ), s1 );
      s2 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), s0 );
      
      _mm_storeu_pd( &B[ 2*j*ldb ], s2 );
	
      s1 = _mm_add_pd( _mm_mul_pd( w2, ah3 ),  _mm_mul_pd( w3, ag3 ) );
      s0 = _mm_add_pd( _mm_mul_pd( w1, ag1 ),  s1 );
      s2 = _mm_add_pd( _mm_mul_pd( w0, ah1 ), s0 );
      
      _mm_storeu_pd( &B[ (2*j+1)*ldb ], s2 );
      
      /* Main loop */
      
      for( i = 1 ; i < N/2 ; i++ ) {
	a0 = _mm_set1_pd( A[ j*lda + i] );
	a1 = _mm_set1_pd( A[ j*lda + N / 2 + i] );
	a2 = _mm_set1_pd( A[ j*lda + i - 1 ] );
	a3 = _mm_set1_pd( A[ j*lda + N / 2 + i - 1 ] );
	
	a4 = _mm_set1_pd( A[ (j+M/2)*lda + i] );
	a5 = _mm_set1_pd( A[ (j+M/2)*lda + N / 2 + i] );
	a6 = _mm_set1_pd( A[ (j+M/2)*lda + i - 1 ] );
	a7 = _mm_set1_pd( A[ (j+M/2)*lda + N / 2 + i - 1 ] );
	
	a8 = _mm_set1_pd( A[ ( j-1 )*lda + i] );
	a9 = _mm_set1_pd( A[ ( j-1 )*lda + N / 2 + i] );
	aa = _mm_set1_pd( A[ ( j-1 )*lda + i - 1 ] );
	ab = _mm_set1_pd( A[ ( j-1 )*lda + N / 2 + i - 1 ] );
	
	ac = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + i] );
	ad = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + N / 2 + i] );
	ae = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + i - 1 ] );
	af = _mm_set1_pd( A[ ( j-1 + M/2 )*lda + N / 2 + i - 1 ] );
	
	/* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
	
	/* W[ j*N + 2*i] */
	w0 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a0, hbegin ), _mm_mul_pd( a1, gbegin ) ),
			 _mm_add_pd( _mm_mul_pd( a2, hend ), _mm_mul_pd( a3, gend ) ) );
	
	/* W[ (j + M/2)*N + i] */
	w1 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a4, hbegin ), _mm_mul_pd( a5, gbegin ) ),
			 _mm_add_pd( _mm_mul_pd( a6, hend ), _mm_mul_pd( a7, gend ) ) );
	
	/* W[((j-1+M/2)%(M/2))*N + i] */
	w2 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( a8, hbegin ), _mm_mul_pd( a9, gbegin ) ),
			 _mm_add_pd( _mm_mul_pd( aa, hend ), _mm_mul_pd( ab, gend ) ) );
	
	/* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
	w3 = _mm_add_pd( _mm_add_pd( _mm_mul_pd( ac, hbegin ), _mm_mul_pd( ad, gbegin ) ),
			 _mm_add_pd( _mm_mul_pd( ae, hend ), _mm_mul_pd( af, gend ) ) );
	
	
	s1 = _mm_add_pd( _mm_mul_pd( w2, ah2 ),  _mm_mul_pd( w3, ag2 ) );
	s0 = _mm_add_pd( _mm_mul_pd( w1, ag0 ), s1 );
	s2 = _mm_add_pd( _mm_mul_pd( w0, ah0 ), s0 );
	
	_mm_storeu_pd( &B[ 2*j*ldb + 2*i], s2 );
	
	s1 = _mm_add_pd( _mm_mul_pd( w2, ah3 ),  _mm_mul_pd( w3, ag3 ) );
	s0 = _mm_add_pd( _mm_mul_pd( w1, ag1 ),  s1 );
	s2 = _mm_add_pd( _mm_mul_pd( w0, ah1 ), s0 );

	_mm_storeu_pd( &B[ (2*j+1)*ldb + 2*i], s2 );
      }
    }
}

#endif // __SSE__ || __aarch64__

#ifdef __AVX__

#ifdef  __cplusplus
void ddi4mt2_avx( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_avx( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, w0, w1, w2, w3, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ) {
            a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
            a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
            a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
            a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            w0 = _mm256_mul_pd( a0, hbegin );
            w1 = _mm256_mul_pd( a1, gbegin );
            w2 = _mm256_mul_pd( a2, hend );
            w3 = _mm256_mul_pd( a3, gend );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ag0 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ag2 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            w0 = _mm256_mul_pd( a0, ah1 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ah3 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );
        }
    }

}

#ifdef  __cplusplus
void ddi4mt2_avx_peel( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_avx_peel( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, w0, w1, w2, w3, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {

      /* Peeling : 1st columns */

      a0 = _mm256_set_pd( A[ j*N + 1 ], A[ j*N + 1 ], A[ j*N ], A[ j*N ] );
      a1 = _mm256_set_pd( A[ j*N + N / 2 + 1 ], A[ j*N + N / 2 + 1 ], A[ j*N + N / 2 ], A[ j*N + N / 2 ] );
      a2 = _mm256_set_pd( A[ j*N ], A[ j*N ], A[ j*N + N/2 - 1 ], A[ j*N + N/2 - 1 ] );
      a3 = _mm256_set_pd( A[ j*N + N / 2 ],  A[ j*N + N / 2 ], A[ j*N + N - 1 ], A[ j*N + N - 1 ] );

      w0 = _mm256_mul_pd( a0, hbegin );
      w1 = _mm256_mul_pd( a1, gbegin );
      w2 = _mm256_mul_pd( a2, hend );
      w3 = _mm256_mul_pd( a3, gend );
      
      s0 = _mm256_add_pd( w0, w1);
      s1 = _mm256_add_pd( w2, w3);
      w = _mm256_add_pd( s0, s1 );
      
      _mm256_storeu_pd( &W[ j*N ], w );
      
      /* Main loop */
      
      for( i = 2 ; i < N / 2 ; i+=2 ) {
	a0 = _mm256_set_pd( A[j*N + i+1], A[j*N + i+1], A[j*N + i], A[j*N + i] );
	a1 = _mm256_set_pd( A[ j*N + N / 2 + i+1], A[ j*N + N / 2 + i+1], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
	a2 = _mm256_set_pd( A[j*N + i ], A[j*N + i ], A[j*N + i - 1 ], A[j*N + i - 1 ] );
	a3 = _mm256_set_pd( A[ j*N + N / 2 + i ],  A[ j*N + N / 2 + i ], A[ j*N + N / 2 + i - 1 ], A[ j*N + N / 2 + i - 1 ] );
	
	w0 = _mm256_mul_pd( a0, hbegin );
	w1 = _mm256_mul_pd( a1, gbegin );
	w2 = _mm256_mul_pd( a2, hend );
	w3 = _mm256_mul_pd( a3, gend );
	
	s0 = _mm256_add_pd( w0, w1);
	s1 = _mm256_add_pd( w2, w3);
	w = _mm256_add_pd( s0, s1 );
	
	_mm256_storeu_pd( &W[ j*N + 2*i], w );
      }
    }

    /* dim 2 */

    /* Peeling: 1st lines */

    for( i = 0 ; i < N ; i+=4 ) {
      a0 = _mm256_loadu_pd( &W[ i] );
      a1 = _mm256_loadu_pd( &W[ ( M/2 )*N + i] );
      a2 = _mm256_loadu_pd( &W[ ( M/2 - 1 )*N + i] );
      a3 = _mm256_loadu_pd( &W[ ( M - 1 )*N + i] );
      
      w0 = _mm256_mul_pd( a0, ah0 );
      w1 = _mm256_mul_pd( a1, ag0 );
      w2 = _mm256_mul_pd( a2, ah2 );
      w3 = _mm256_mul_pd( a3, ag2 );
      
      s0 = _mm256_add_pd( w0, w1);
      s1 = _mm256_add_pd( w2, w3);
      w = _mm256_add_pd( s0, s1 );
      
      _mm256_storeu_pd( &B[ i], w );
      
      w0 = _mm256_mul_pd( a0, ah1 );
      w1 = _mm256_mul_pd( a1, ag1 );
      w2 = _mm256_mul_pd( a2, ah3 );
      w3 = _mm256_mul_pd( a3, ag3 );

      s0 = _mm256_add_pd( w0, w1);
      s1 = _mm256_add_pd( w2, w3);
      w = _mm256_add_pd( s0, s1 );
      
      _mm256_storeu_pd( &B[ ldb + i], w );
    }

    /* Main loop */
    
    for( j = 1 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ag0 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ag2 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            w0 = _mm256_mul_pd( a0, ah1 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ah3 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

	    _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );
        }
    }

}

#ifdef  __cplusplus
void ddi4mt2_avx_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_avx_gather( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, w0, w1, w2, w3, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;
    __m256i stride = _mm256_set_epi64x( sizeof( double ), sizeof( double ),
                                          0, 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 2 ; i < N / 2 - 2 ; i+=2 ) {
            a0 = _mm256_i64gather_pd( &A[j*lda + i], stride, 1 );
            a1 = _mm256_i64gather_pd( &A[j*lda + lda / 2 + i], stride, 1 );
            a2 = _mm256_i64gather_pd( &A[j*lda + ( i - 1 + lda/2 ) % (lda/2)], stride, 1 );
            a3 = _mm256_i64gather_pd( &A[ j*lda + lda / 2 + ( i - 1 + lda/2) % ( lda / 2 )], stride, 1 );

            w0 = _mm256_mul_pd( a0, hbegin );
            w1 = _mm256_mul_pd( a1, gbegin );
            w2 = _mm256_mul_pd( a2, hend );
            w3 = _mm256_mul_pd( a3, gend );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* The last column cannot be done with gather because of the folding */

    for( j = 0 ; j < M ; j++ ) {
        i = 0;
        a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
        a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
        a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
        a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

        w0 = _mm256_mul_pd( a0, hbegin );
        w1 = _mm256_mul_pd( a1, gbegin );
        w2 = _mm256_mul_pd( a2, hend );
        w3 = _mm256_mul_pd( a3, gend );

        s0 = _mm256_add_pd( w0, w1);
        s1 = _mm256_add_pd( w2, w3);
        w = _mm256_add_pd( s0, s1 );

        _mm256_storeu_pd( &W[ j*N + 2*i], w );

        i = N/2 - 2;
        a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
        a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
        a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
        a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

        w0 = _mm256_mul_pd( a0, hbegin );
        w1 = _mm256_mul_pd( a1, gbegin );
        w2 = _mm256_mul_pd( a2, hend );
        w3 = _mm256_mul_pd( a3, gend );

        s0 = _mm256_add_pd( w0, w1);
        s1 = _mm256_add_pd( w2, w3);
        w = _mm256_add_pd( s0, s1 );

        _mm256_storeu_pd( &W[ j*N + 2*i], w );
    }

   /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ag0 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ag2 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            w0 = _mm256_mul_pd( a0, ah1 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ah3 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );

        }
    }

}
#endif // __AVX__

#ifdef __AVX2__

#ifdef  __cplusplus
void ddi4mt2_fma( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );


    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ) {
            a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
            a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
            a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
            a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */

            s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
            s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );

            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ag0 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ag2 ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a0, ah1,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ah3,  _mm256_mul_pd( a3, ag3 ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );
        }
    }

}

#ifdef  __cplusplus
void ddi4mt2_fma_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma_gather( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;
    __m256i stride = _mm256_set_epi64x( sizeof( double ), sizeof( double ),
                                          0, 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 2 ; i < N / 2 - 2; i+=2 ) {
            a0 = _mm256_i64gather_pd( &A[j*lda + i], stride, 1 );
            a1 = _mm256_i64gather_pd( &A[j*lda + lda / 2 + i], stride, 1 );
            a2 = _mm256_i64gather_pd( &A[j*lda + ( i - 1 + lda/2 ) % (lda/2)], stride, 1 );
            a3 = _mm256_i64gather_pd( &A[ j*lda + lda / 2 + ( i - 1 + lda/2) % ( lda / 2 )], stride, 1 );

            s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
            s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* The last column cannot be done with gather because of the folding */

    for( j = 0 ; j < M ; j++ ) {
        i = 0;
        a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
        a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
        a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
        a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

        s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
        s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );
        w = _mm256_add_pd( s0, s1 );

        _mm256_storeu_pd( &W[ j*N + 2*i], w );

        i = N/2 - 2;
        a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
        a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
        a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
        a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

        s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
        s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );
        w = _mm256_add_pd( s0, s1 );

        _mm256_storeu_pd( &W[ j*N + 2*i], w );
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ag0 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ag2 ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a0, ah1,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ah3,  _mm256_mul_pd( a3, ag3 ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );

        }
    }
}

#ifdef  __cplusplus
void ddi4mt2_fma2( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma2( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );


    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ) {
            a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
            a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
            a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
            a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

             /* Variant:
                w = a0 * hbegin + ( a1 * gbegin + ( a2 * hend + (a3 * gend))) */

            s0 = _mm256_fmadd_pd( a2, hend, _mm256_mul_pd( a3, gend ) );
            s1 = _mm256_fmadd_pd( a1, gbegin, s0 );
            w = _mm256_fmadd_pd( a0, hbegin, s1 );

            _mm256_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ag2 ) );
            s0 = _mm256_fmadd_pd( a1, ag0, s1 );
            w = _mm256_fmadd_pd( a0, ah0, s0 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            s1 = _mm256_fmadd_pd( a2, ah3,  _mm256_mul_pd( a3, ag3 ) );
            s0 = _mm256_fmadd_pd( a1, ag1,  s1 );
            w = _mm256_fmadd_pd( a0, ah1, s0 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );
        }
    }
}

#ifdef  __cplusplus
void ddi4mt2_fma2_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma2_gather( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

     __m256d w, s0, s1;
     __m256d a0, a1, a2, a3;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;
    __m256i stride = _mm256_set_epi64x( sizeof( double ), sizeof( double ),
                                          0, 0 );

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 2 ; i < N / 2 - 2 ; i+=2 ) {
            a0 = _mm256_i64gather_pd( &A[j*lda + i], stride, 1 );
            a1 = _mm256_i64gather_pd( &A[j*N + N / 2 + i], stride, 1 );
            a2 = _mm256_i64gather_pd( &A[j*N + ( i - 1 + N/2 ) % (N/2)], stride, 1 );
            a3 = _mm256_i64gather_pd( &A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], stride, 1 );

             /* Variant:
                w = a0 * hbegin + ( a1 * gbegin + ( a2 * hend + (a3 * gend))) */

            s0 = _mm256_fmadd_pd( a2, hend, _mm256_mul_pd( a3, gend ) );
            s1 = _mm256_fmadd_pd( a1, gbegin, s0 );
            w = _mm256_fmadd_pd( a0, hbegin, s1 );

            _mm256_storeu_pd( &W[ j*N + 2*i], w );
        }
    }

    /* The last column cannot be done with gather because of the folding */

    for( j = 0 ; j < M ; j++ ) {
        i = 0;
        a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
        a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
        a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
        a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

        s0 = _mm256_fmadd_pd( a2, hend, _mm256_mul_pd( a3, gend ) );
        s1 = _mm256_fmadd_pd( a1, gbegin, s0 );
        w = _mm256_fmadd_pd( a0, hbegin, s1 );

        _mm256_storeu_pd( &W[ j*N + 2*i], w );

        i = N/2 - 2;
        a0 = _mm256_set_pd( A[j*N + (i+1)], A[j*N + (i+1)], A[j*N + i], A[j*N + i] );
        a1 = _mm256_set_pd( A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + (i+1)], A[ j*N + N / 2 + i], A[ j*N + N / 2 + i] );
        a2 = _mm256_set_pd( A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*N + ( i - 1 + N/2 ) % (N/2) ], A[j*N + ( i - 1 + N/2 ) % (N/2) ] );
        a3 = _mm256_set_pd(  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*N + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

        s0 = _mm256_fmadd_pd( a2, hend, _mm256_mul_pd( a3, gend ) );
        s1 = _mm256_fmadd_pd( a1, gbegin, s0 );
        w = _mm256_fmadd_pd( a0, hbegin, s1 );

        _mm256_storeu_pd( &W[ j*N + 2*i], w );
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[ j*N + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*N + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*N + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*N + i] );

            s0 = _mm256_fmadd_pd( a0, ah0,  _mm256_mul_pd( a1, ag0 ) );
            s1 = _mm256_fmadd_pd( a2, ah2,  _mm256_mul_pd( a3, ag2 ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );

            s0 = _mm256_fmadd_pd( a0, ah1,  _mm256_mul_pd( a1, ag1 ) );
            s1 = _mm256_fmadd_pd( a2, ah3,  _mm256_mul_pd( a3, ag3 ) );
            w = _mm256_add_pd( s0, s1 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );
        }
    }
}

 #ifdef  __cplusplus
void ddi4mt2_fma_reuse( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma_reuse( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d w0, w1, w2, w3, s0, s1, s2, s3, s4, s5, s6, s7;
     __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );


    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N/2 ; i+=2 ) {
            a0 = _mm256_set_pd( A[j*lda + (i+1)], A[j*lda + (i+1)], A[j*lda + i], A[j*lda + i] );
            a1 = _mm256_set_pd( A[ j*lda + N / 2 + (i+1)], A[ j*lda + N / 2 + (i+1)], A[ j*lda + N / 2 + i], A[ j*lda + N / 2 + i] );
            a2 = _mm256_set_pd( A[j*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[j*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*lda + ( i - 1 + N/2 ) % (N/2) ], A[j*lda + ( i - 1 + N/2 ) % (N/2) ] );
            a3 = _mm256_set_pd(  A[ j*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            a4 = _mm256_set_pd( A[(j+M/2)*lda + (i+1)], A[(j+M/2)*lda + (i+1)], A[(j+M/2)*lda + i], A[(j+M/2)*lda + i] );
            a5 = _mm256_set_pd( A[ (j+M/2)*lda + N / 2 + (i+1)], A[ (j+M/2)*lda + N / 2 + (i+1)], A[ (j+M/2)*lda + N / 2 + i], A[ (j+M/2)*lda + N / 2 + i] );
            a6 = _mm256_set_pd( A[(j+M/2)*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[(j+M/2)*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[(j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ], A[(j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
            a7 = _mm256_set_pd(  A[ (j+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ (j+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            a8 = _mm256_set_pd( A[((j-1+M/2)%(M/2))*lda + (i+1)], A[((j-1+M/2)%(M/2))*lda + (i+1)], A[((j-1+M/2)%(M/2))*lda + i], A[((j-1+M/2)%(M/2))*lda + i] );
            a9 = _mm256_set_pd( A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+1)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+1)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i] );
            a10 = _mm256_set_pd( A[((j-1+M/2)%(M/2))*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[((j-1+M/2)%(M/2))*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ], A[((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ] );
            a11 = _mm256_set_pd(  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            a12 = _mm256_set_pd( A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+1)], A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+1)], A[(( j-1+M/2)%(M/2)+M/2)*lda + i], A[(( j-1+M/2)%(M/2)+M/2)*lda + i] );
            a13 = _mm256_set_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+1)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+1)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i] );
            a14 = _mm256_set_pd( A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
            a15 = _mm256_set_pd(  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */

            s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
            s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );

            s2 = _mm256_fmadd_pd( a4, hbegin,  _mm256_mul_pd( a5, gbegin ) );
            s3 = _mm256_fmadd_pd( a6, hend,  _mm256_mul_pd( a7, gend ) );

            s4 = _mm256_fmadd_pd( a8, hbegin,  _mm256_mul_pd( a9, gbegin ) );
            s5 = _mm256_fmadd_pd( a10, hend,  _mm256_mul_pd( a11, gend ) );

            s6 = _mm256_fmadd_pd( a12, hbegin,  _mm256_mul_pd( a13, gbegin ) );
            s7 = _mm256_fmadd_pd( a14, hend,  _mm256_mul_pd( a15, gend ) );

            /* W[ j*N + 2*i] */
            w0 = _mm256_add_pd( s0, s1 );
            /* W[ (j + M/2)*N + i] */
            w1 = _mm256_add_pd( s2, s3 );
            /* W[((j-1+M/2)%(M/2))*N + i] */
            w2 = _mm256_add_pd( s4, s5 );
            /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
            w3 = _mm256_add_pd( s6, s7 );

            s1 = _mm256_fmadd_pd( w2, ah2,  _mm256_mul_pd( w3, ag2 ) );
            s0 = _mm256_fmadd_pd( w1, ag0, s1 );
            s3 = _mm256_fmadd_pd( w0, ah0, s0 );

            _mm256_storeu_pd( &B[ 2*j*ldb + 2*i], s3 );

            s1 = _mm256_fmadd_pd( w2, ah3,  _mm256_mul_pd( w3, ag3 ) );
            s0 = _mm256_fmadd_pd( w1, ag1,  s1 );
            s3 = _mm256_fmadd_pd( w0, ah1, s0 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + 2*i], s3 );

        }
    }

}

#ifdef  __cplusplus
void ddi4mt2_fma_reuse_peel( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma_reuse_peel( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m256d w0, w1, w2, w3, s0, s1, s2, s3, s4, s5, s6, s7;
     __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
    __m256d ah0, ah1, ah2, ah3;
    __m256d ag0, ag1, ag2, ag3;
    __m256d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm256_set_pd( h1, h0, h1, h0);
    gbegin = _mm256_set_pd( g1, g0, g1, g0 );
    hend = _mm256_set_pd( h3, h2, h3, h2);
    gend = _mm256_set_pd( g3, g2, g3, g2 );
    ah0 = _mm256_set1_pd( h0 );
    ah1 = _mm256_set1_pd( h1 );
    ah2 = _mm256_set1_pd( h2 );
    ah3 = _mm256_set1_pd( h3 );
    ag0 = _mm256_set1_pd( g0 );
    ag1 = _mm256_set1_pd( g1 );
    ag2 = _mm256_set1_pd( g2 );
    ag3 = _mm256_set1_pd( g3 );

    /* Peeling: top left corner */

    a0 = _mm256_set_pd( A[ 1 ], A[ 1 ], A[ 0 ], A[ 0 ] );
    a1 = _mm256_set_pd( A[ N / 2 + 1 ], A[ N / 2 + 1 ], A[ N / 2 ], A[ N / 2 ] );
    a2 = _mm256_set_pd( A[ 0 ], A[ 0 ], A[ N / 2 - 1 ], A[ N / 2 - 1 ] );
    a3 = _mm256_set_pd( A[ N / 2 ],  A[ N / 2 ], A[ N - 1 ], A[ N - 1 ] );

    a4 = _mm256_set_pd( A[ (M/2)*lda + 1 ], A[ (M/2)*lda + 1 ], A[ (M/2)*lda ], A[ (M/2)*lda ] );
    a5 = _mm256_set_pd( A[ (M/2)*lda + N / 2 + 1 ], A[ (M/2)*lda + N / 2 + 1 ], A[ (M/2)*lda + N / 2 ], A[ (M/2)*lda + N / 2 ] );
    a6 = _mm256_set_pd( A[ (M/2)*lda ], A[ (M/2)*lda ], A[ (M/2)*lda + N / 2 - 1 ], A[ (M/2)*lda + N / 2 - 1 ] );
    a7 = _mm256_set_pd( A[ (M/2)*lda + N / 2 ], A[ (M/2)*lda + N / 2 ], A[ (M/2)*lda + N - 1 ], A[ (M/2)*lda + N - 1 ] );

    a8 = _mm256_set_pd( A[ (M/2 - 1)*lda + 1], A[ (M/2 - 1)*lda + 1 ], A[ (M/2 - 1)*lda ], A[ (M/2 - 1)*lda ] );
    a9 = _mm256_set_pd( A[ (M/2 - 1)*lda + N / 2 + 1 ], A[ (M/2 - 1)*lda + N / 2 + 1 ], A[ (M/2 - 1)*lda + N / 2 ], A[ ( M/2 - 1)*lda + N / 2 ] );
    a10 = _mm256_set_pd( A[ (M/2 - 1)*lda ], A[ (M/2 - 1)*lda ], A[ (M/2 - 1)*lda + N / 2 - 1 ], A[ (M/2 - 1)*lda + N / 2 - 1 ] );
    a11 = _mm256_set_pd( A[ (M/2 - 1)*lda + N / 2 ],  A[ (M/2 - 1)*lda + N / 2 ], A[ (M/2 - 1)*lda + N - 1 ], A[ (M/2 - 1)*lda + N - 1 ] );

    a12 = _mm256_set_pd( A[ ( M - 1 )*lda + 1 ], A[ ( M - 1 )*lda + 1 ], A[ ( M - 1 )*lda ], A[ ( M - 1 )*lda ] );
    a13 = _mm256_set_pd( A[ ( M - 1 )*lda + N / 2 + 1 ], A[ ( M - 1 )*lda + N / 2 + 1 ], A[ ( M - 1 )*lda + N / 2 ], A[ ( M - 1 )*lda + N / 2 ] );
    a14 = _mm256_set_pd( A[ ( M - 1 )*lda ], A[ ( M - 1 )*lda ], A[ ( M - 1 )*lda + N/2 - 1 ], A[ ( M - 1 )*lda + N/2 - 1 ] );
    a15 = _mm256_set_pd( A[ ( M - 1 )*lda + N / 2 ],  A[ ( M - 1 )*lda + N / 2 ], A[ ( M - 1 )*lda + N - 1 ], A[ ( M - 1 )*lda + N - 1 ] );
    
    /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
    
    s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
    s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );
    
    s2 = _mm256_fmadd_pd( a4, hbegin,  _mm256_mul_pd( a5, gbegin ) );
    s3 = _mm256_fmadd_pd( a6, hend,  _mm256_mul_pd( a7, gend ) );
    
    s4 = _mm256_fmadd_pd( a8, hbegin,  _mm256_mul_pd( a9, gbegin ) );
    s5 = _mm256_fmadd_pd( a10, hend,  _mm256_mul_pd( a11, gend ) );
    
    s6 = _mm256_fmadd_pd( a12, hbegin,  _mm256_mul_pd( a13, gbegin ) );
    s7 = _mm256_fmadd_pd( a14, hend,  _mm256_mul_pd( a15, gend ) );
    
    /* W[ j*N + 2*i] */
    w0 = _mm256_add_pd( s0, s1 );
    /* W[ (j + M/2)*N + i] */
    w1 = _mm256_add_pd( s2, s3 );
    /* W[((j-1+M/2)%(M/2))*N + i] */
    w2 = _mm256_add_pd( s4, s5 );
    /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
    w3 = _mm256_add_pd( s6, s7 );
    
    s1 = _mm256_fmadd_pd( w2, ah2,  _mm256_mul_pd( w3, ag2 ) );
    s0 = _mm256_fmadd_pd( w1, ag0, s1 );
    s3 = _mm256_fmadd_pd( w0, ah0, s0 );
    
    _mm256_storeu_pd( &B[ 0 ], s3 );
    
    s1 = _mm256_fmadd_pd( w2, ah3,  _mm256_mul_pd( w3, ag3 ) );
    s0 = _mm256_fmadd_pd( w1, ag1,  s1 );
    s3 = _mm256_fmadd_pd( w0, ah1, s0 );
    
    _mm256_storeu_pd( &B[ ldb], s3 );
    
    /* Peeling: first line */

    for( i = 2 ; i < N/2 ; i+=2 ) {

      a0 = _mm256_set_pd( A[ i+1 ], A[ i+1 ], A[ i ], A[ i ] );
      a1 = _mm256_set_pd( A[ N/2 + i + 1 ], A[ N/2 + i + 1 ], A[ N/2 + i ], A[ N/2 + i ] );
      a2 = _mm256_set_pd( A[ i ], A[ i ], A[ i - 1 ], A[ i - 1 ] );
      a3 = _mm256_set_pd( A[ N/2 + i ], A[ N/2 + i ], A[ N/2 + i - 1 ], A[ N/2 + i - 1 ] );
      
      a4 = _mm256_set_pd( A[ ( M/2 )*lda + i+1 ], A[ ( M/2 )*lda + i+1 ], A[ ( M/2 )*lda + i ], A[ ( M/2 )*lda + i ] );
      a5 = _mm256_set_pd( A[ ( M/2 )*lda + N/2 + i + 1 ], A[ ( M/2 )*lda + N/2 + i + 1 ], A[ ( M/2 )*lda + N/2 + i ], A[ ( M/2 )*lda + N/2 + i ] );
      a6 = _mm256_set_pd( A[ ( M/2 )*lda + i ], A[ ( M/2 )*lda + i ], A[ ( M/2 )*lda + i - 1 ], A[ ( M/2 )*lda + i - 1 ] );
      a7 = _mm256_set_pd( A[ ( M/2 )*lda + N/2 + i ], A[ ( M/2 )*lda + N/2 + i ], A[ ( M/2 )*lda + N/2 + i - 1 ], A[ ( M/2 )*lda + N/2 + i - 1 ] );
      
      a8 = _mm256_set_pd( A[ ( M/2 - 1 )*lda + i+1 ], A[ ( M/2 - 1 )*lda + i+1 ], A[ ( M/2 - 1 )*lda + i ], A[ ( M/2 - 1 )*lda + i ] );
      a9 = _mm256_set_pd( A[ ( M/2 - 1 )*lda + N/2 + i + 1 ], A[ ( M/2 - 1 )*lda + N/2 + i + 1 ], A[ ( M/2 - 1 )*lda + N/2 + i ], A[ ( M/2 - 1 )*lda + N/2 + i ] );
      a10 = _mm256_set_pd( A[ ( M/2 - 1 )*lda + i ], A[ ( M/2 - 1 )*lda + i ], A[ ( M/2 - 1 )*lda + i - 1 ], A[ ( M/2 - 1 )*lda + i - 1 ] );
      a11 = _mm256_set_pd( A[ ( M/2 - 1 )*lda + N/2 + i ], A[ ( M/2 - 1 )*lda + N/2 + i ], A[ ( M/2 - 1 )*lda + N/2 + i - 1 ], A[ ( M/2 - 1 )*lda + N/2 + i - 1 ] );
      
      a12 = _mm256_set_pd( A[ ( M - 1 )*lda + i+1 ], A[ ( M - 1 )*lda + i+1 ], A[ ( M - 1 )*lda + i ], A[ ( M - 1 )*lda + i ] );
      a13 = _mm256_set_pd( A[ ( M - 1 )*lda + N/2 + i + 1 ], A[ ( M - 1 )*lda + N/2 + i + 1 ], A[ ( M - 1 )*lda + N/2 + i ], A[ ( M - 1 )*lda + N/2 + i ] );
      a14 = _mm256_set_pd( A[ ( M - 1 )*lda + i ], A[ ( M - 1 )*lda + i ], A[ ( M - 1 )*lda + i - 1 ], A[ ( M - 1 )*lda + i - 1 ] );
      a15 = _mm256_set_pd( A[ ( M - 1 )*lda + N/2 + i ], A[ ( M - 1 )*lda + N/2 + i ], A[ ( M - 1 )*lda + N/2 + i - 1 ], A[ ( M - 1 )*lda + N/2 + i - 1 ] ); 

      /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
      
      s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
      s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );
      
      s2 = _mm256_fmadd_pd( a4, hbegin,  _mm256_mul_pd( a5, gbegin ) );
      s3 = _mm256_fmadd_pd( a6, hend,  _mm256_mul_pd( a7, gend ) );
      
      s4 = _mm256_fmadd_pd( a8, hbegin,  _mm256_mul_pd( a9, gbegin ) );
      s5 = _mm256_fmadd_pd( a10, hend,  _mm256_mul_pd( a11, gend ) );
      
      s6 = _mm256_fmadd_pd( a12, hbegin,  _mm256_mul_pd( a13, gbegin ) );
      s7 = _mm256_fmadd_pd( a14, hend,  _mm256_mul_pd( a15, gend ) );
      
      /* W[ j*N + 2*i] */
      w0 = _mm256_add_pd( s0, s1 );
      /* W[ (j + M/2)*N + i] */
      w1 = _mm256_add_pd( s2, s3 );
      /* W[((j-1+M/2)%(M/2))*N + i] */
      w2 = _mm256_add_pd( s4, s5 );
      /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
      w3 = _mm256_add_pd( s6, s7 );
      
      s1 = _mm256_fmadd_pd( w2, ah2,  _mm256_mul_pd( w3, ag2 ) );
      s0 = _mm256_fmadd_pd( w1, ag0, s1 );
      s3 = _mm256_fmadd_pd( w0, ah0, s0 );
      
      _mm256_storeu_pd( &B[ 2*i], s3 );
      
      s1 = _mm256_fmadd_pd( w2, ah3,  _mm256_mul_pd( w3, ag3 ) );
      s0 = _mm256_fmadd_pd( w1, ag1,  s1 );
      s3 = _mm256_fmadd_pd( w0, ah1, s0 );
      
      _mm256_storeu_pd( &B[ ldb + 2*i], s3 );
      
    }

    for( j = 1 ; j < M/2 ; j++ ) {

      /* Peeling: first column */

      a0 =  _mm256_set_pd( A[ j*lda + 1 ], A[ j*lda + 1 ], A[ j*lda ], A[ j*lda ]);
      a1 = _mm256_set_pd( A[ j*lda + N / 2 + 1 ], A[ j*lda + N / 2 + 1 ], A[ j*lda + N / 2 ], A[ j*lda + N / 2 ]);
      a2 = _mm256_set_pd( A[ j*lda ], A[ j*lda ], A[ j*lda + N / 2 - 1 ], A[ j*lda + N / 2 - 1 ]);
      a3 = _mm256_set_pd( A[ j*lda + N / 2 ], A[ j*lda + N / 2 ], A[ j*lda + N - 1 ], A[ j*lda + N - 1 ]);
      
      a4 = _mm256_set_pd( A[ (j+M/2)*lda + 1 ], A[ (j+M/2)*lda + 1 ], A[ (j+M/2)*lda ], A[ (j+M/2)*lda ]);
      a5 = _mm256_set_pd( A[ (j+M/2)*lda + N / 2 + 1 ], A[ (j+M/2)*lda + N / 2 + 1 ], A[ (j+M/2)*lda + N / 2 ], A[ (j+M/2)*lda + N / 2 ]);
      a6 = _mm256_set_pd( A[ (j+M/2)*lda ], A[ (j+M/2)*lda ], A[ (j+M/2)*lda + N / 2 - 1 ], A[ (j+M/2)*lda + N / 2 - 1 ]);
      a7 = _mm256_set_pd( A[ (j+M/2)*lda + N / 2 ], A[ (j+M/2)*lda + N / 2 ], A[ (j+M/2)*lda + N - 1 ], A[ (j+M/2)*lda + N - 1 ]);
      
      a8 = _mm256_set_pd( A[ (j-1)*lda + 1 ], A[ (j-1)*lda + 1 ], A[ (j-1)*lda ], A[ (j-1)*lda ]);
      a9 = _mm256_set_pd( A[ (j-1)*lda + N / 2 + 1 ], A[ (j-1)*lda + N / 2 + 1 ], A[ (j-1)*lda + N / 2 ], A[ (j-1)*lda + N / 2 ]);
      a10 = _mm256_set_pd( A[ (j-1)*lda ], A[ (j-1)*lda ], A[ (j-1)*lda + N / 2 - 1 ], A[ (j-1)*lda + N / 2 - 1 ]);
      a11 = _mm256_set_pd( A[ (j-1)*lda + N / 2 ], A[ (j-1)*lda + N / 2 ], A[ (j-1)*lda + N - 1 ], A[ (j-1)*lda + N - 1 ]);
      
      a12 = _mm256_set_pd( A[ (j-1+M/2)*lda + 1 ], A[ (j-1+M/2)*lda + 1 ], A[ (j-1+M/2)*lda ], A[ (j-1+M/2)*lda ]);
      a13 = _mm256_set_pd( A[ (j-1+M/2)*lda + N / 2 + 1 ], A[ (j-1+M/2)*lda + N / 2 + 1 ], A[ (j-1+M/2)*lda + N / 2 ], A[ (j-1+M/2)*lda + N / 2 ]);
      a14 = _mm256_set_pd( A[ (j-1+M/2)*lda ], A[ (j-1+M/2)*lda ], A[ (j-1+M/2)*lda + N / 2 - 1 ], A[ (j-1+M/2)*lda + N / 2 - 1 ]);
      a15 = _mm256_set_pd( A[ (j-1+M/2)*lda + N / 2 ], A[ (j-1+M/2)*lda + N / 2 ], A[ (j-1+M/2)*lda + N - 1 ], A[ (j-1+M/2)*lda + N - 1 ]);

      /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */
	
      s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
      s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );
      
      s2 = _mm256_fmadd_pd( a4, hbegin,  _mm256_mul_pd( a5, gbegin ) );
      s3 = _mm256_fmadd_pd( a6, hend,  _mm256_mul_pd( a7, gend ) );
      
      s4 = _mm256_fmadd_pd( a8, hbegin,  _mm256_mul_pd( a9, gbegin ) );
      s5 = _mm256_fmadd_pd( a10, hend,  _mm256_mul_pd( a11, gend ) );
      
      s6 = _mm256_fmadd_pd( a12, hbegin,  _mm256_mul_pd( a13, gbegin ) );
      s7 = _mm256_fmadd_pd( a14, hend,  _mm256_mul_pd( a15, gend ) );
      
      /* W[ j*N + 2*i] */
      w0 = _mm256_add_pd( s0, s1 );
      /* W[ (j + M/2)*N + i] */
      w1 = _mm256_add_pd( s2, s3 );
      /* W[((j-1+M/2)%(M/2))*N + i] */
      w2 = _mm256_add_pd( s4, s5 );
      /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
      w3 = _mm256_add_pd( s6, s7 );
      
      s1 = _mm256_fmadd_pd( w2, ah2,  _mm256_mul_pd( w3, ag2 ) );
      s0 = _mm256_fmadd_pd( w1, ag0, s1 );
      s3 = _mm256_fmadd_pd( w0, ah0, s0 );
      
      _mm256_storeu_pd( &B[ 2*j*ldb ], s3 );
      
      s1 = _mm256_fmadd_pd( w2, ah3,  _mm256_mul_pd( w3, ag3 ) );
      s0 = _mm256_fmadd_pd( w1, ag1,  s1 );
      s3 = _mm256_fmadd_pd( w0, ah1, s0 );
      
      _mm256_storeu_pd( &B[ (2*j+1)*ldb], s3 );
      

      /* Main loop */
      
      for( i = 2 ; i < N/2 ; i+=2 ) {

	a0 = _mm256_set_pd( A[ j*lda + i+1 ], A[ j*lda + i+1 ], A[ j*lda + i ], A[ j*lda + i ] );
a1 = _mm256_set_pd( A[ j*lda + N/2 + i + 1 ], A[ j*lda + N/2 + i + 1 ], A[ j*lda + N/2 + i ], A[ j*lda + N/2 + i ] );
a2 = _mm256_set_pd( A[ j*lda + i ], A[ j*lda + i ], A[ j*lda + i - 1 ], A[ j*lda + i - 1 ] );
a3 = _mm256_set_pd( A[ j*lda + N/2 + i ], A[ j*lda + N/2 + i ], A[ j*lda + N/2 + i - 1 ], A[ j*lda + N/2 + i - 1 ] );
a4 = _mm256_set_pd( A[ (j+M/2)*lda + i+1 ], A[ (j+M/2)*lda + i+1 ], A[ (j+M/2)*lda + i ], A[ (j+M/2)*lda + i ] );
a5 = _mm256_set_pd( A[ (j+M/2)*lda + N/2 + i + 1 ], A[ (j+M/2)*lda + N/2 + i + 1 ], A[ (j+M/2)*lda + N/2 + i ], A[ (j+M/2)*lda + N/2 + i ] );
a6 = _mm256_set_pd( A[ (j+M/2)*lda + i ], A[ (j+M/2)*lda + i ], A[ (j+M/2)*lda + i - 1 ], A[ (j+M/2)*lda + i - 1 ] );
a7 = _mm256_set_pd( A[ (j+M/2)*lda + N/2 + i ], A[ (j+M/2)*lda + N/2 + i ], A[ (j+M/2)*lda + N/2 + i - 1 ], A[ (j+M/2)*lda + N/2 + i - 1 ] );
a8 = _mm256_set_pd( A[ (j-1)*lda + i+1 ], A[ (j-1)*lda + i+1 ], A[ (j-1)*lda + i ], A[ (j-1)*lda + i ] );
a9 = _mm256_set_pd( A[ (j-1)*lda + N/2 + i + 1 ], A[ (j-1)*lda + N/2 + i + 1 ], A[ (j-1)*lda + N/2 + i ], A[ (j-1)*lda + N/2 + i ] );
a10 = _mm256_set_pd( A[ (j-1)*lda + i ], A[ (j-1)*lda + i ], A[ (j-1)*lda + i - 1 ], A[ (j-1)*lda + i - 1 ] );
a11 = _mm256_set_pd( A[ (j-1)*lda + N/2 + i ], A[ (j-1)*lda + N/2 + i ], A[ (j-1)*lda + N/2 + i - 1 ], A[ (j-1)*lda + N/2 + i - 1 ] );
a12 = _mm256_set_pd( A[ (j-1+M/2)*lda + i+1 ], A[ (j-1+M/2)*lda + i+1 ], A[ (j-1+M/2)*lda + i ], A[ (j-1+M/2)*lda + i ] );
a13 = _mm256_set_pd( A[ (j-1+M/2)*lda + N/2 + i + 1 ], A[ (j-1+M/2)*lda + N/2 + i + 1 ], A[ (j-1+M/2)*lda + N/2 + i ], A[ (j-1+M/2)*lda + N/2 + i ] );
a14 = _mm256_set_pd( A[ (j-1+M/2)*lda + i ], A[ (j-1+M/2)*lda + i ], A[ (j-1+M/2)*lda + i - 1 ], A[ (j-1+M/2)*lda + i - 1 ] );
a15 = _mm256_set_pd( A[ (j-1+M/2)*lda + N/2 + i ], A[ (j-1+M/2)*lda + N/2 + i ], A[ (j-1+M/2)*lda + N/2 + i - 1 ], A[ (j-1+M/2)*lda + N/2 + i - 1 ] );

	/*
	a0 = _mm256_set_pd( A[j*lda + (i+1)], A[j*lda + (i+1)], A[j*lda + i], A[j*lda + i] );
	a1 = _mm256_set_pd( A[ j*lda + N / 2 + (i+1)], A[ j*lda + N / 2 + (i+1)], A[ j*lda + N / 2 + i], A[ j*lda + N / 2 + i] );
	a2 = _mm256_set_pd( A[j*lda + i ], A[j*lda + i ], A[j*lda + i - 1 ], A[j*lda + i - 1 ] );
	a3 = _mm256_set_pd( A[ j*lda + N / 2 + i ],  A[ j*lda + N / 2 + i ], A[ j*lda + N / 2 + i - 1 ], A[ j*lda + N / 2 + i - 1 ] );
	
	a4 = _mm256_set_pd( A[(j+M/2)*lda + (i+1)], A[(j+M/2)*lda + (i+1)], A[(j+M/2)*lda + i], A[(j+M/2)*lda + i] );
	a5 = _mm256_set_pd( A[ (j+M/2)*lda + N / 2 + (i+1)], A[ (j+M/2)*lda + N / 2 + (i+1)], A[ (j+M/2)*lda + N / 2 + i], A[ (j+M/2)*lda + N / 2 + i] );
	a6 = _mm256_set_pd( A[(j+M/2)*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[(j+M/2)*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[(j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ], A[(j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
	a7 = _mm256_set_pd(  A[ (j+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ (j+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );
	
	a8 = _mm256_set_pd( A[((j-1+M/2)%(M/2))*lda + (i+1)], A[((j-1+M/2)%(M/2))*lda + (i+1)], A[((j-1+M/2)%(M/2))*lda + i], A[((j-1+M/2)%(M/2))*lda + i] );
	a9 = _mm256_set_pd( A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+1)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+1)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i] );
	a10 = _mm256_set_pd( A[((j-1+M/2)%(M/2))*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[((j-1+M/2)%(M/2))*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ], A[((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ] );
	a11 = _mm256_set_pd(  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );
	
	a12 = _mm256_set_pd( A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+1)], A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+1)], A[(( j-1+M/2)%(M/2)+M/2)*lda + i], A[(( j-1+M/2)%(M/2)+M/2)*lda + i] );
	a13 = _mm256_set_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+1)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+1)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i] );
	a14 = _mm256_set_pd( A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
	a15 = _mm256_set_pd(  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );
	*/

            /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */

            s0 = _mm256_fmadd_pd( a0, hbegin,  _mm256_mul_pd( a1, gbegin ) );
            s1 = _mm256_fmadd_pd( a2, hend,  _mm256_mul_pd( a3, gend ) );

            s2 = _mm256_fmadd_pd( a4, hbegin,  _mm256_mul_pd( a5, gbegin ) );
            s3 = _mm256_fmadd_pd( a6, hend,  _mm256_mul_pd( a7, gend ) );

            s4 = _mm256_fmadd_pd( a8, hbegin,  _mm256_mul_pd( a9, gbegin ) );
            s5 = _mm256_fmadd_pd( a10, hend,  _mm256_mul_pd( a11, gend ) );

            s6 = _mm256_fmadd_pd( a12, hbegin,  _mm256_mul_pd( a13, gbegin ) );
            s7 = _mm256_fmadd_pd( a14, hend,  _mm256_mul_pd( a15, gend ) );

            /* W[ j*N + 2*i] */
            w0 = _mm256_add_pd( s0, s1 );
            /* W[ (j + M/2)*N + i] */
            w1 = _mm256_add_pd( s2, s3 );
            /* W[((j-1+M/2)%(M/2))*N + i] */
            w2 = _mm256_add_pd( s4, s5 );
            /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
            w3 = _mm256_add_pd( s6, s7 );

            s1 = _mm256_fmadd_pd( w2, ah2,  _mm256_mul_pd( w3, ag2 ) );
            s0 = _mm256_fmadd_pd( w1, ag0, s1 );
            s3 = _mm256_fmadd_pd( w0, ah0, s0 );

            _mm256_storeu_pd( &B[ 2*j*ldb + 2*i], s3 );

            s1 = _mm256_fmadd_pd( w2, ah3,  _mm256_mul_pd( w3, ag3 ) );
            s0 = _mm256_fmadd_pd( w1, ag1,  s1 );
            s3 = _mm256_fmadd_pd( w0, ah1, s0 );

            _mm256_storeu_pd( &B[ (2*j+1)*ldb + 2*i], s3 );
	    
      }
}
 
}
 
#endif // __AVX2__

#ifdef __AVX512F__

 #ifdef  __cplusplus
void ddi4mt2_fma512_reuse( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma512_reuse( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j;

    __m512d w0, w1, w2, w3, s0, s1, s2, s3, s4, s5, s6, s7;
     __m512d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
    __m512d ah0, ah1, ah2, ah3;
    __m512d ag0, ag1, ag2, ag3;
    __m512d hbegin, hend, gbegin, gend;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;
    hbegin = _mm512_set_pd( h1, h0, h1, h0, h1, h0, h1, h0 );
    gbegin = _mm512_set_pd( g1, g0, g1, g0, g1, g0, g1, g0 );
    hend = _mm512_set_pd( h3, h2, h3, h2, h3, h2, h3, h2 );
    gend = _mm512_set_pd( g3, g2, g3, g2, g3, g2, g3, g2 );
    ah0 = _mm512_set1_pd( h0 );
    ah1 = _mm512_set1_pd( h1 );
    ah2 = _mm512_set1_pd( h2 );
    ah3 = _mm512_set1_pd( h3 );
    ag0 = _mm512_set1_pd( g0 );
    ag1 = _mm512_set1_pd( g1 );
    ag2 = _mm512_set1_pd( g2 );
    ag3 = _mm512_set1_pd( g3 );


    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N/2 ; i+=4 ) {
	  a0 = _mm512_set_pd( A[j*lda + (i+3)], A[j*lda + (i+3)], A[j*lda + i + 2], A[j*lda + i + 2 ], A[j*lda + (i+1)], A[j*lda + (i+1)], A[j*lda + i], A[j*lda + i] );
	  a1 = _mm512_set_pd( A[ j*lda + N / 2 + (i+3)], A[ j*lda + N / 2 + (i+3)], A[ j*lda + N / 2 + i + 2 ], A[ j*lda + N / 2 + i + 2 ],  A[ j*lda + N / 2 + (i+1)], A[ j*lda + N / 2 + (i+1)], A[ j*lda + N / 2 + i], A[ j*lda + N / 2 + i] );
	  a2 = _mm512_set_pd( A[j*lda + ( (i+3) - 1 + N/2 ) % (lda/2)], A[j*lda + ( (i+3) - 1 + N/2 ) % (N/2)], A[j*lda + ( ( i + 2 ) - 1 + N/2 ) % (N/2) ], A[j*lda + ( ( i + 2 ) - 1 + N/2 ) % (N/2) ], A[j*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[j*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[j*lda + ( i - 1 + N/2 ) % (N/2) ], A[j*lda + ( i - 1 + N/2 ) % (N/2) ] );
	  a3 = _mm512_set_pd(  A[ j*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))],  A[ j*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))], A[ j*lda + N / 2 + ( ( i + 2 ) - 1 + N/2) % ( N / 2 )], A[ j*lda + N / 2 + ( ( i + 2 ) - 1 + N/2) % ( N / 2 )],  A[ j*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ j*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ j*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ j*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

	  a4 = _mm512_set_pd( A[(j+M/2)*lda + (i+3)], A[(j+M/2)*lda + (i+3)], A[(j+M/2)*lda + (i+2)], A[(j+M/2)*lda + (i+2)], A[(j+M/2)*lda + (i+1)], A[(j+M/2)*lda + (i+1)], A[(j+M/2)*lda + i], A[(j+M/2)*lda + i] );
	  a5 = _mm512_set_pd( A[ (j+M/2)*lda + N / 2 + (i+3)], A[ (j+M/2)*lda + N / 2 + (i+3)], A[ (j+M/2)*lda + N / 2 + (i+2)], A[ (j+M/2)*lda + N / 2 + (i+2)],  A[ (j+M/2)*lda + N / 2 + (i+1)], A[ (j+M/2)*lda + N / 2 + (i+1)], A[ (j+M/2)*lda + N / 2 + i], A[ (j+M/2)*lda + N / 2 + i] );
	  a6 = _mm512_set_pd( A[(j+M/2)*lda + ( (i+3) - 1 + N/2 ) % (lda/2)], A[(j+M/2)*lda + ( (i+3) - 1 + N/2 ) % (N/2)], A[(j+M/2)*lda + ( (i+2) - 1 + N/2 ) % (N/2) ], A[(j+M/2)*lda + ( (i+2) - 1 + N/2 ) % (N/2) ],  A[(j+M/2)*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[(j+M/2)*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[(j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ], A[(j+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
	  a7 = _mm512_set_pd(  A[ (j+M/2)*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))],  A[ (j+M/2)*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))], A[ (j+M/2)*lda + N / 2 + ( (i+2) - 1 + N/2) % ( N / 2 )], A[ (j+M/2)*lda + N / 2 + ( (i+2) - 1 + N/2) % ( N / 2 )],  A[ (j+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ (j+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ (j+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

	  a8 = _mm512_set_pd( A[((j-1+M/2)%(M/2))*lda + (i+3)], A[((j-1+M/2)%(M/2))*lda + (i+3)], A[((j-1+M/2)%(M/2))*lda + (i+2)], A[((j-1+M/2)%(M/2))*lda + (i+2)],  A[((j-1+M/2)%(M/2))*lda + (i+1)], A[((j-1+M/2)%(M/2))*lda + (i+1)], A[((j-1+M/2)%(M/2))*lda + i], A[((j-1+M/2)%(M/2))*lda + i] );
	  a9 = _mm512_set_pd( A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+3)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+3)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+2)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+2)] ,  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+1)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + (i+1)], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + i] );
  	  a10 = _mm512_set_pd( A[((j-1+M/2)%(M/2))*lda + ( (i+3) - 1 + N/2 ) % (lda/2)], A[((j-1+M/2)%(M/2))*lda + ( (i+3) - 1 + N/2 ) % (N/2)], A[((j-1+M/2)%(M/2))*lda + ( (i+2) - 1 + N/2 ) % (N/2) ], A[((j-1+M/2)%(M/2))*lda + ( (i+2) - 1 + N/2 ) % (N/2) ], A[((j-1+M/2)%(M/2))*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[((j-1+M/2)%(M/2))*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ], A[((j-1+M/2)%(M/2))*lda + ( i - 1 + N/2 ) % (N/2) ] );
	  a11 = _mm512_set_pd(  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))],  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( (i+2) - 1 + N/2) % ( N / 2 )], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( (i+2) - 1 + N/2) % ( N / 2 )],  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ ((j-1+M/2)%(M/2))*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

	  a12 = _mm512_set_pd( A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+3)], A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+3)], A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+2)],  A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+1)], A[(( j-1+M/2)%(M/2)+M/2)*lda + (i+1)], A[(( j-1+M/2)%(M/2)+M/2)*lda + i], A[(( j-1+M/2)%(M/2)+M/2)*lda + i] );
	  a13 = _mm512_set_pd( A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+3)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+3)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+2)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+2)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+1)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + (i+1)], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + i] );
	  a14 = _mm512_set_pd( A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+3) - 1 + N/2 ) % (lda/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+3) - 1 + N/2 ) % (N/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+2) - 1 + N/2 ) % (N/2) ], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+2) - 1 + N/2 ) % (N/2) ], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+1) - 1 + N/2 ) % (lda/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( (i+1) - 1 + N/2 ) % (N/2)], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ], A[(( j-1+M/2)%(M/2)+M/2)*lda + ( i - 1 + N/2 ) % (N/2) ] );
	  a15 = _mm512_set_pd(  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))],  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+3) - 1 + N/2) %(  N/2))], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( (i+2) - 1 + N/2) % ( N / 2 )], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( (i+2) - 1 + N/2) % ( N / 2 )], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))],  A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( ( (i+1) - 1 + N/2) %(  N/2))], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )], A[ (( j-1+M/2)%(M/2)+M/2)*lda + N / 2 + ( i - 1 + N/2) % ( N / 2 )] );

            /* w = ( a0 * hbegin + ( a1 * gbegin ) ) + ( a2 * ( hend + (a3 * gend))) */

            s0 = _mm512_fmadd_pd( a0, hbegin,  _mm512_mul_pd( a1, gbegin ) );
            s1 = _mm512_fmadd_pd( a2, hend,  _mm512_mul_pd( a3, gend ) );

            s2 = _mm512_fmadd_pd( a4, hbegin,  _mm512_mul_pd( a5, gbegin ) );
            s3 = _mm512_fmadd_pd( a6, hend,  _mm512_mul_pd( a7, gend ) );

            s4 = _mm512_fmadd_pd( a8, hbegin,  _mm512_mul_pd( a9, gbegin ) );
            s5 = _mm512_fmadd_pd( a10, hend,  _mm512_mul_pd( a11, gend ) );

            s6 = _mm512_fmadd_pd( a12, hbegin,  _mm512_mul_pd( a13, gbegin ) );
            s7 = _mm512_fmadd_pd( a14, hend,  _mm512_mul_pd( a15, gend ) );

            /* W[ j*N + 2*i] */
            w0 = _mm512_add_pd( s0, s1 );
            /* W[ (j + M/2)*N + i] */
            w1 = _mm512_add_pd( s2, s3 );
            /* W[((j-1+M/2)%(M/2))*N + i] */
            w2 = _mm512_add_pd( s4, s5 );
            /* W[ (( j-1+M/2)%(M/2)+M/2)*N + i] */
            w3 = _mm512_add_pd( s6, s7 );

            s1 = _mm512_fmadd_pd( w2, ah2,  _mm512_mul_pd( w3, ag2 ) );
            s0 = _mm512_fmadd_pd( w1, ag0, s1 );
            s3 = _mm512_fmadd_pd( w0, ah0, s0 );

            _mm512_storeu_pd( &B[ 2*j*ldb + 2*i], s3 );

            s1 = _mm512_fmadd_pd( w2, ah3,  _mm512_mul_pd( w3, ag3 ) );
            s0 = _mm512_fmadd_pd( w1, ag1,  s1 );
            s3 = _mm512_fmadd_pd( w0, ah1, s0 );

            _mm512_storeu_pd( &B[ (2*j+1)*ldb + 2*i], s3 );

        }
    }

}
#endif // __AVX512F__
