/* This file is part of pawt.
 *
 * Various implementations of the 2D Daubechies D6 transform dda6mt2.
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

#include "daub6.h"


/*
c     Compute 2D Daubechies D6 transform of a matrix
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
c     da6 Daubechies D6
c     mt matrix transform
c     2 2D
*/

#ifdef  __cplusplus
void dda6mt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda6mt2_initial( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h[6], g[6];
    int i, j, k;

    dGetCoeffs6( h, g );

    /* dim 1 */

    for( k = 0 ; k < M ; k++ ) {
        i = j = 0;
        while( i < N-1 ) {
            W[ k*N + j] = h[0] * A[k*lda + i] + h[1] * A[ k*lda + i+1] + h[2] * A[k*lda + ( i + 2 ) % N ] + h[3] * A[ k*lda + ( i + 3 ) % N ]  + h[4] * A[k*lda + ( i + 4 ) % N ] + h[5] * A[ k*lda + ( i + 5 ) % N ];
            i += 2;
            j++;
        }
        i = 0;
        while( i < N-1 ) {
            W[ k*N + j] = g[0] * A[k*lda + i] + g[1] * A[ k*lda + i+1] + g[2] * A[k*lda + ( i + 2 ) % N ] + g[3] * A[ k*lda + ( i + 3 ) % N ]  + g[4] * A[k*lda + ( i + 4 ) % N ] + g[5] * A[ k*lda + ( i + 5 ) % N ];
            i += 2;
            j++;
        }
    }

    /* dim 2 */

    for( k = 0, i = 0 ; k < M ; k+=2, i++ ) { /* upper half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] = h[0] * W[k*N + j] + h[1] * W[(k+1)*N + j] + h[2] * W[((k+2)%M)*N + j] + h[3] * W[((k+3)%M)*N + j] + h[4] * W[((k+4)%M)*N + j] + h[5] * W[((k+5)%M)*N + j];
        }
    }
    for( k = 0, i = M/2; k < M ; k+=2, i++ ) { /* lower half */
        for( j = 0 ; j < N ; j++ ) {
            B[i*ldb + j] = g[0] * W[k*N + j] + g[1] * W[(k+1)*N + j] + g[2] * W[((k+2)%M)*N + j] + g[3] * W[((k+3)%M)*N + j] + g[4] * W[((k+4)%M)*N + j] + g[5] * W[((k+5)%M)*N + j];
        }
    }
}

 #ifdef  __cplusplus
void dda6mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda6mt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h[6], g[6];
    int i, j;

    dGetCoeffs6( h, g );

    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < (N / 2) ; i++ ){
            W[ j*N + i] = h[0] * A[j*lda + 2*i] + h[1] * A[ j*lda + 2*i+1] + h[2] * A[j*lda + ( 2*i + 2 ) % N ] + h[3] * A[ j*lda + ( 2*i + 3 ) % N ]  + h[4] * A[j*lda + ( 2*i + 4 ) % N ] + h[5] * A[ j*lda + ( 2*i + 5 ) % N ];
            W[ j*N + i + N/2] = g[0] * A[j*lda + 2*i] + g[1] * A[ j*lda + 2*i+1] + g[2] * A[j*lda + ( 2*i + 2 ) % N ] + g[3] * A[ j*lda + ( 2*i + 3 ) % N ]  + g[4] * A[j*lda + ( 2*i + 4 ) % N ] + g[5] * A[ j*lda + ( 2*i + 5 ) % N ];
        }
    }

    /* dim 2 */

    for( j = 0 ; j < (M / 2) ; j++ ){
        for( i = 0 ; i < N ; i++ ){
            B[i + j * ldb] = h[0] * W[2*j*N + i] + h[1] * W[(2*j+1)*N + i] + h[2] * W[((2*j+2)%M)*N + i] + h[3] * W[((2*j+3)%M)*N + i] + h[4] * W[((2*j+4)%M)*N + i] + h[5] * W[((2*j+5)%M)*N + i];
            B[i + (j + N/2)*ldb] = g[0] * W[2*j*N + i] + g[1] * W[(2*j+1)*N + i] + g[2] * W[((2*j+2)%M)*N + i] + g[3] * W[((2*j+3)%M)*N + i] + g[4] * W[((2*j+4)%M)*N + i] + g[5] * W[((2*j+5)%M)*N + i];

        }
    }
}

#ifdef  __cplusplus
void dda6mt2_avx( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dda6mt2_avx( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif

  double h[6], g[6];
  int i, j;

  dGetCoeffs6( h, g );

  __m256d a0, a1, a2, a3, a4, a5;
  __m256d w0, w1, w2, w3, w4, w5, s0, s1, s2;
  __m256d ah0, ah1, ah2, ah3, ah4, ah5;
  __m256d ag0, ag1, ag2, ag3, ag4, ag5;

  ah0 = _mm256_set1_pd( h[0] );
  ah1 = _mm256_set1_pd( h[1] );
  ah2 = _mm256_set1_pd( h[2] );
  ah3 = _mm256_set1_pd( h[3] );
  ah4 = _mm256_set1_pd( h[4] );
  ah5 = _mm256_set1_pd( h[5] );
  ag0 = _mm256_set1_pd( g[0] );
  ag1 = _mm256_set1_pd( g[1] );
  ag2 = _mm256_set1_pd( g[2] );
  ag3 = _mm256_set1_pd( g[3] );
  ag4 = _mm256_set1_pd( g[4] );
  ag5 = _mm256_set1_pd( g[5] );

  /* Dim 1 */

  for( j = 0 ; j < M ; j++ ) {
  	for( i = 0 ; i < N / 2 -  2  ; i+= 4 ){
      a0  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  0  )], A[ j*lda + ( 2*(i +  2 ) +  0  )], A[ j*lda + ( 2*(i +  1 ) +  0  )], A[ j*lda + ( 2*(i +  0 ) +  0  )] );
      a1  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  1  )], A[ j*lda + ( 2*(i +  2 ) +  1  )], A[ j*lda + ( 2*(i +  1 ) +  1  )], A[ j*lda + ( 2*(i +  0 ) +  1  )] );
      a2  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  2  )], A[ j*lda + ( 2*(i +  2 ) +  2  )], A[ j*lda + ( 2*(i +  1 ) +  2  )], A[ j*lda + ( 2*(i +  0 ) +  2  )] );
      a3  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  3  )], A[ j*lda + ( 2*(i +  2 ) +  3  )], A[ j*lda + ( 2*(i +  1 ) +  3  )], A[ j*lda + ( 2*(i +  0 ) +  3  )] );
      a4  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  4  )], A[ j*lda + ( 2*(i +  2 ) +  4  )], A[ j*lda + ( 2*(i +  1 ) +  4  )], A[ j*lda + ( 2*(i +  0 ) +  4  )] );
      a5  = _mm256_set_pd( A[ j*lda + ( 2*(i +  3 ) +  5  )], A[ j*lda + ( 2*(i +  2 ) +  5  )], A[ j*lda + ( 2*(i +  1 ) +  5  )], A[ j*lda + ( 2*(i +  0 ) +  5  )] );

      w0 = _mm256_mul_pd( a0, ah0 );
      w1 = _mm256_mul_pd( a1, ah1 );
      w2 = _mm256_mul_pd( a2, ah2 );
      w3 = _mm256_mul_pd( a3, ah3 );
      w4 = _mm256_mul_pd( a4, ah4 );
      w5 = _mm256_mul_pd( a5, ah5 );

      s0 = _mm256_add_pd( w0, w1 );
      s1 = _mm256_add_pd( w2, w3 );
      s2 = _mm256_add_pd( w4, w5 );
      s0 = _mm256_add_pd( s0, s1 );
      s0 = _mm256_add_pd( s0, s2 );

      _mm256_storeu_pd( &W[ j*ldb + i ], s0 );

      w0 = _mm256_mul_pd( a0, ag0 );
      w1 = _mm256_mul_pd( a1, ag1 );
      w2 = _mm256_mul_pd( a2, ag2 );
      w3 = _mm256_mul_pd( a3, ag3 );
      w4 = _mm256_mul_pd( a4, ag4 );
      w5 = _mm256_mul_pd( a5, ag5 );

      s0 = _mm256_add_pd( w0, w1 );
      s1 = _mm256_add_pd( w2, w3 );
      s2 = _mm256_add_pd( w4, w5 );
      s0 = _mm256_add_pd( s0, s1 );
      s0 = _mm256_add_pd( s0, s2 );

      _mm256_storeu_pd( &W[ j*ldb + i + N/2], s0 );
  	}
  }

  /* Peeling */

  for( j = 0 ; j < M ; j++ ) {
    a0  = _mm256_set_pd( A[ j*lda + N - 2 ] , A[ j*lda + N - 4 ] , A[ j*lda + N - 6 ] , A[ j*lda + N - 8 ]  );
    a1  = _mm256_set_pd( A[ j*lda + N - 1 ] , A[ j*lda + N - 3 ] , A[ j*lda + N - 5 ] , A[ j*lda + N - 7 ]  );
    a2  = _mm256_set_pd( A[ j*lda + 0 ], A[ j*lda + N - 2 ] , A[ j*lda + N - 4 ] , A[ j*lda + N - 6 ]  );
    a3  = _mm256_set_pd( A[ j*lda + 1 ], A[ j*lda + N - 1 ] , A[ j*lda + N - 3 ] , A[ j*lda + N - 5 ]  );
    a4  = _mm256_set_pd( A[ j*lda + 2 ], A[ j*lda + 0 ], A[ j*lda + N - 2 ] , A[ j*lda + N - 4 ]  );
    a5  = _mm256_set_pd( A[ j*lda + 3 ], A[ j*lda + 1 ], A[ j*lda + N - 1 ] , A[ j*lda + N - 3 ]  );

    w0 = _mm256_mul_pd( a0, ah0 );
    w1 = _mm256_mul_pd( a1, ah1 );
    w2 = _mm256_mul_pd( a2, ah2 );
    w3 = _mm256_mul_pd( a3, ah3 );
    w4 = _mm256_mul_pd( a4, ah4 );
    w5 = _mm256_mul_pd( a5, ah5 );

    s0 = _mm256_add_pd( w0, w1 );
    s1 = _mm256_add_pd( w2, w3 );
    s2 = _mm256_add_pd( w4, w5 );
    s0 = _mm256_add_pd( s0, s1 );
    s0 = _mm256_add_pd( s0, s2 );

    _mm256_storeu_pd( &W[ j*ldb + N/2 - 4 ], s0 );

    w0 = _mm256_mul_pd( a0, ag0 );
    w1 = _mm256_mul_pd( a1, ag1 );
    w2 = _mm256_mul_pd( a2, ag2 );
    w3 = _mm256_mul_pd( a3, ag3 );
    w4 = _mm256_mul_pd( a4, ag4 );
    w5 = _mm256_mul_pd( a5, ag5 );

    s0 = _mm256_add_pd( w0, w1 );
    s1 = _mm256_add_pd( w2, w3 );
    s2 = _mm256_add_pd( w4, w5 );
    s0 = _mm256_add_pd( s0, s1 );
    s0 = _mm256_add_pd( s0, s2 );

    _mm256_storeu_pd( &W[ j*ldb + N - 4 ], s0 );
  }


  /* Dim 2 */

  for( j = 0 ; j < M / 2 - 2  ; j++ ) {
  	for( i = 0 ; i < N ; i+= 4 ){
      a0  = _mm256_loadu_pd( &W[( 2*j +0 ) * N + i] );
      a1  = _mm256_loadu_pd( &W[( 2*j +1 ) * N + i] );
      a2  = _mm256_loadu_pd( &W[( 2*j +2 ) * N + i] );
      a3  = _mm256_loadu_pd( &W[( 2*j +3 ) * N + i] );
      a4  = _mm256_loadu_pd( &W[( 2*j +4 ) * N + i] );
      a5  = _mm256_loadu_pd( &W[( 2*j +5 ) * N + i] );

      w0 = _mm256_mul_pd( a0, ah0 );
      w1 = _mm256_mul_pd( a1, ah1 );
      w2 = _mm256_mul_pd( a2, ah2 );
      w3 = _mm256_mul_pd( a3, ah3 );
      w4 = _mm256_mul_pd( a4, ah4 );
      w5 = _mm256_mul_pd( a5, ah5 );

      s0 = _mm256_add_pd( w0, w1 );
      s1 = _mm256_add_pd( w2, w3 );
      s2 = _mm256_add_pd( w4, w5 );
      s0 = _mm256_add_pd( s0, s1 );
      s0 = _mm256_add_pd( s0, s2 );

      _mm256_storeu_pd( &B[ j*ldb + i], s0 );

      w0 = _mm256_mul_pd( a0, ag0 );
      w1 = _mm256_mul_pd( a1, ag1 );
      w2 = _mm256_mul_pd( a2, ag2 );
      w3 = _mm256_mul_pd( a3, ag3 );
      w4 = _mm256_mul_pd( a4, ag4 );
      w5 = _mm256_mul_pd( a5, ag5 );

      s0 = _mm256_add_pd( w0, w1 );
      s1 = _mm256_add_pd( w2, w3 );
      s2 = _mm256_add_pd( w4, w5 );
      s0 = _mm256_add_pd( s0, s1 );
      s0 = _mm256_add_pd( s0, s2 );

      _mm256_storeu_pd( &B[ (j+M/2)*ldb + i], s0 );
  	}
  }

  /* Peeling */

  for( i = 0 ; i < N ; i+= 4 ){
    a0  = _mm256_loadu_pd( &W[ ( N -  4 )* N + i] );
    a1  = _mm256_loadu_pd( &W[ ( N -  3 )* N + i] );
    a2  = _mm256_loadu_pd( &W[ ( N -  2 )* N + i] );
    a3  = _mm256_loadu_pd( &W[ ( N -  1 )* N + i] );
    a4  = _mm256_loadu_pd( &W[ i ] );
    a5  = _mm256_loadu_pd( &W[ (1 )* N + i] );

    w0 = _mm256_mul_pd( a0, ah0 );
    w1 = _mm256_mul_pd( a1, ah1 );
    w2 = _mm256_mul_pd( a2, ah2 );
    w3 = _mm256_mul_pd( a3, ah3 );
    w4 = _mm256_mul_pd( a4, ah4 );
    w5 = _mm256_mul_pd( a5, ah5 );

    s0 = _mm256_add_pd( w0, w1 );
    s1 = _mm256_add_pd( w2, w3 );
    s2 = _mm256_add_pd( w4, w5 );
    s0 = _mm256_add_pd( s0, s1 );
    s0 = _mm256_add_pd( s0, s2 );

    _mm256_storeu_pd( &B[ (M/2 - 2)*ldb + i], s0 );

    w0 = _mm256_mul_pd( a0, ag0 );
    w1 = _mm256_mul_pd( a1, ag1 );
    w2 = _mm256_mul_pd( a2, ag2 );
    w3 = _mm256_mul_pd( a3, ag3 );
    w4 = _mm256_mul_pd( a4, ag4 );
    w5 = _mm256_mul_pd( a5, ag5 );

    s0 = _mm256_add_pd( w0, w1 );
    s1 = _mm256_add_pd( w2, w3 );
    s2 = _mm256_add_pd( w4, w5 );
    s0 = _mm256_add_pd( s0, s1 );
    s0 = _mm256_add_pd( s0, s2 );

    _mm256_storeu_pd( &B[ (M - 2)*ldb + i], s0 );
  }

	for( i = 0 ; i < N ; i+= 4 ){
    a0  = _mm256_loadu_pd( &W[ ( N -  2 )* N + i] );
    a1  = _mm256_loadu_pd( &W[ ( N -  1 )* N + i] );
    a2  = _mm256_loadu_pd( &W[  i] );
    a3  = _mm256_loadu_pd( &W[ ( 1 )* N + i] );
    a4  = _mm256_loadu_pd( &W[ (2 )* N + i] );
    a5  = _mm256_loadu_pd( &W[ (3 )* N + i] );

    w0 = _mm256_mul_pd( a0, ah0 );
    w1 = _mm256_mul_pd( a1, ah1 );
    w2 = _mm256_mul_pd( a2, ah2 );
    w3 = _mm256_mul_pd( a3, ah3 );
    w4 = _mm256_mul_pd( a4, ah4 );
    w5 = _mm256_mul_pd( a5, ah5 );

    s0 = _mm256_add_pd( w0, w1 );
    s1 = _mm256_add_pd( w2, w3 );
    s2 = _mm256_add_pd( w4, w5 );
    s0 = _mm256_add_pd( s0, s1 );
    s0 = _mm256_add_pd( s0, s2 );

    _mm256_storeu_pd( &B[ (M/2 - 1)*ldb + i], s0 );

    w0 = _mm256_mul_pd( a0, ag0 );
    w1 = _mm256_mul_pd( a1, ag1 );
    w2 = _mm256_mul_pd( a2, ag2 );
    w3 = _mm256_mul_pd( a3, ag3 );
    w4 = _mm256_mul_pd( a4, ag4 );
    w5 = _mm256_mul_pd( a5, ag5 );

    s0 = _mm256_add_pd( w0, w1 );
    s1 = _mm256_add_pd( w2, w3 );
    s2 = _mm256_add_pd( w4, w5 );
    s0 = _mm256_add_pd( s0, s1 );
    s0 = _mm256_add_pd( s0, s2 );

    _mm256_storeu_pd( &B[ (M - 1)*ldb + i], s0 );
  }
}
