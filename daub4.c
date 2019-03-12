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

#include <x86intrin.h>
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
    int i, j, k;

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


#ifdef  __cplusplus
void ddi4mt2_avx( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_avx( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j, k;
    
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
          
            _mm256_storeu_pd( &W[ j*N + i], w );             
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[2*j*lda + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*lda + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*lda + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*lda + i] ); 
            
            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ah1 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ah3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );
            
            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );             

            w0 = _mm256_mul_pd( a0, ag0 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ag2 );
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
    int i, j, k;
    
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

    
    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ) {
            a0 = _mm256_i64gather_pd( &A[j*lda + i], stride, 2 );
            a1 = _mm256_i64gather_pd( &A[j*N + N / 2 + i], stride, 2 );
            a2 = _mm256_i64gather_pd( &A[j*N + ( i - 1 + N/2 ) % (N/2)], stride, 2 );
            a3 = _mm256_i64gather_pd( &A[ j*N + N / 2 + ( i - 1 + N/2) % ( N / 2 )], stride, 2 );

            w0 = _mm256_mul_pd( a0, hbegin );
            w1 = _mm256_mul_pd( a1, gbegin );
            w2 = _mm256_mul_pd( a2, hend );
            w3 = _mm256_mul_pd( a3, gend );
        
            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );
          
            _mm256_storeu_pd( &W[ j*N + i], w );             
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[2*j*lda + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*lda + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*lda + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*lda + i] ); 
            
            w0 = _mm256_mul_pd( a0, ah0 );
            w1 = _mm256_mul_pd( a1, ah1 );
            w2 = _mm256_mul_pd( a2, ah2 );
            w3 = _mm256_mul_pd( a3, ah3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );
            
            _mm256_storeu_pd( &B[ 2*j*ldb + i], w );             

            w0 = _mm256_mul_pd( a0, ag0 );
            w1 = _mm256_mul_pd( a1, ag1 );
            w2 = _mm256_mul_pd( a2, ag2 );
            w3 = _mm256_mul_pd( a3, ag3 );

            s0 = _mm256_add_pd( w0, w1);
            s1 = _mm256_add_pd( w2, w3);
            w = _mm256_add_pd( s0, s1 );
            
            _mm256_storeu_pd( &B[ (2*j+1)*ldb + i], w );
            
        }
    }
    
}

#ifdef  __cplusplus
void ddi4mt2_fma( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void ddi4mt2_fma( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    double h0, h1, h2, h3;
    double g0, g1, g2, g3;
    int i, j, k;
    
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
          
            _mm256_storeu_pd( &W[ j*N + i], w );             
        }
    }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ) {
        for( i = 0 ; i < N ; i+=4 ) {
            a0 = _mm256_loadu_pd( &W[2*j*lda + i] );
            a1 = _mm256_loadu_pd( &W[ (j + M/2)*lda + i] );
            a2 = _mm256_loadu_pd( &W[((j-1+M/2)%(M/2))*lda + i] );
            a3 = _mm256_loadu_pd( &W[ (( j-1+M/2)%(M/2)+M/2)*lda + i] ); 
            
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
