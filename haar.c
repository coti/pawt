/* This file is part of pawt.
 *
 * Various implementations of the 2D Haar transform dhamt2.
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

#include <string.h>

#include <x86intrin.h>

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

#ifdef  __cplusplus
void dhamt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dhamt2_initial( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
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

#ifdef  __cplusplus
void dhamt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dhamt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    int i, j;

    /* dim 1 */
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i++ ){ 
            W[ j*ldb + i] = ( A[j*lda + 2*i] + A[ j*lda + 2*i+1] ) / 2.0;
        }
        for( i = 0 ; i < N / 2 ; i++ ){ 
            W[ j*ldb + i + N/2] = ( A[j*lda + 2*i] - A[ j*lda + 2*i+1] ) / 2.0;
        }
    }
    
    /* dim 2 */
    
    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i++ ){
            B[i + j * ldb] = ( W[ i+ 2*j*lda ] + W[ i+ (2*j+1)*lda] ) / 2.0;
            B[i + (j+M/2) * ldb ] = ( W[ i+ 2*j*lda ] - W[ i + (2*j+1)*lda] ) / 2.0;
        }
    }
        
}

/* TODO handle case when the matrix is too small */
     
void dhamt2_avx( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w, a1, a2;
    const __m256d deux = _mm256_set1_pd( 0.5 );

    /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    /* dim 1 */
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){ 
            a1 = _mm256_set_pd( A[j*lda + 2*i + 6], A[j*lda + 2*i + 4], A[j*lda + 2*i + 2], A[j*lda + 2*i] );
            a2 = _mm256_set_pd( A[j*lda + 2*i + 7], A[j*lda + 2*i + 5], A[j*lda + 2*i + 3], A[j*lda + 2*i + 1] );
            
            w = _mm256_add_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &W[ j*ldb + i], w );             

            w = _mm256_sub_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &W[ j*ldb + i + N/2], w ); 

        }
    }
    
    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i+=4 ){
            a1 = _mm256_loadu_pd(  &W[ 2* j*lda + i] );
            a2 =  _mm256_loadu_pd(  &W[ ( 2 * j + 1 ) *lda + i] );
            w = _mm256_add_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &B[ j*ldb + i ], w ); 
            
            w = _mm256_sub_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i ], w ); 
        }
    }
    
}

void dhamt2_avx_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w, a1, a2;
    const __m256d deux = _mm256_set1_pd( 0.5 );
    __m256i stride =   _mm256_set_epi64x( 3*sizeof( double ), 2*sizeof( double ),
                                          sizeof( double ), 0 );
     /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    /* dim 1 */
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){ 
            a1 = _mm256_i64gather_pd( &A[j*lda + 2*i], stride, 2 );
            a2 = _mm256_i64gather_pd( &A[j*lda + 2*i + 1], stride, 2 );
            w = _mm256_add_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &W[ j*ldb + i], w );             
        }
        for( i = 0 ; i < N / 2 ; i+=4 ){ 
            a1 = _mm256_i64gather_pd( &A[j*lda + 2*i], stride, 2 );
            a2 = _mm256_i64gather_pd( &A[j*lda + 2*i + 1], stride, 2 );
            w = _mm256_sub_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &W[ j*ldb + i + N/2], w ); 
       }
    }
    
    /* dim 2 */
    
    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i+=4 ){
            a1 = _mm256_loadu_pd(  &W[ 2* j*lda + i] );
            a2 =  _mm256_loadu_pd(  &W[ ( 2 * j + 1 ) *lda + i] );
            w = _mm256_add_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &B[ j*ldb + i ], w ); 
            
            w = _mm256_sub_pd( a1, a2 );
            w = _mm256_mul_pd( w, deux );
            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i ], w ); 
        }
    }
    
}
 
void dhamt2_sse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
     int i, j;
     __m128d w, a1, a2;
    const __m128d deux = _mm_set1_pd( 0.5 );
    
     /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    /* dim 1 */

    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ){ 
            a1 = _mm_set_pd( A[j*lda + 2*i + 2], A[j*lda + 2*i] );
            a2 = _mm_set_pd( A[j*lda + 2*i + 3], A[j*lda + 2*i + 1] );
            w = _mm_add_pd( a1, a2 );
            w = _mm_mul_pd( w, deux );
            _mm_storeu_pd( &W[ j*ldb + i], w );             
        }
         for( i = 0 ; i < N / 2 ; i+=2 ){ 
            a1 = _mm_set_pd( A[j*lda + 2*i + 2], A[j*lda + 2*i] );
            a2 = _mm_set_pd( A[j*lda + 2*i + 3], A[j*lda + 2*i + 1] );
            w = _mm_sub_pd( a1, a2 );
            w = _mm_mul_pd( w, deux );
            _mm_storeu_pd( &W[ j*ldb + i + N/2], w ); 
       }
   }

    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i+=2 ){
            a1 = _mm_load_pd( &W[ 2 * j * lda + i] );
            a2 = _mm_load_pd( &W[ ( 2 * j + 1 ) * lda + i ] );
            w = _mm_add_pd( a1, a2 );
            w = _mm_mul_pd( w, deux );
            _mm_storeu_pd( &B[ j*ldb + i ], w ); 

            w = _mm_sub_pd( a1, a2 );
            w = _mm_mul_pd( w, deux );
            _mm_storeu_pd( &B[ (j+M/2)*ldb + i ], w ); 
        }
    }

 }
 
/* TODO handle case when the matrix is too small */
     
void dhamt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w, a1, a2;
    const __m256d deux = _mm256_set1_pd( 0.5 );
    const __m256d moinsdeux = _mm256_set1_pd( -0.5 );

    /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    /* dim 1 */
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){ 
            a1 = _mm256_set_pd( A[j*lda + 2*i + 6], A[j*lda + 2*i + 4], A[j*lda + 2*i + 2], A[j*lda + 2*i] );
            a2 = _mm256_set_pd( A[j*lda + 2*i + 7], A[j*lda + 2*i + 5], A[j*lda + 2*i + 3], A[j*lda + 2*i + 1] );

            w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, deux ) );
            _mm256_storeu_pd( &W[ j*ldb + i], w );             
            
            w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, moinsdeux ) );
            _mm256_storeu_pd( &W[ j*ldb + i + N/2], w ); 
        }
    }
    
    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i+=4 ){
            a1 = _mm256_loadu_pd(  &W[ 2* j*lda + i] );
            a2 =  _mm256_loadu_pd(  &W[ ( 2 * j + 1 ) *lda + i] );

            w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, deux ) );
            _mm256_storeu_pd( &B[ j*ldb + i ], w ); 
            
            w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, moinsdeux ) );
            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i ], w ); 
        }
    }
    
}

#define BLOCKl 4
#define BLOCKc 512
 
void dhamt2_fma_block( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j, a, b;
    __m256d w, a1, a2;
    const __m256d deux = _mm256_set1_pd( 0.5 );
    const __m256d moinsdeux = _mm256_set1_pd( -0.5 );

    /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    /* dim 1 */

    /* First half */

    for( a = 0 ; a < M ; a += BLOCKl ){
        for( b = 0 ; b < N/2 ; b += BLOCKc ){
        
            for( j = 0 ; j < BLOCKl ; j++ ) {
                for( i = 0 ; i < BLOCKc ; i+=4 ){
                    
                    a1 = _mm256_set_pd( A[(a+j)*lda + 2*(b+i) + 6], A[(a+j)*lda + 2*(b+i) + 4], A[(a+j)*lda + 2*(b+i) + 2], A[(a+j)*lda + 2*(b+i)] );
                    a2 = _mm256_set_pd( A[(a+j)*lda + 2*(b+i) + 7], A[(a+j)*lda + 2*(b+i) + 5], A[(a+j)*lda + 2*(b+i) + 3], A[(a+j)*lda + 2*(b+i) + 1] );
                    
                    w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, deux ) );
                    _mm256_storeu_pd( &W[ j*ldb + (b+i)], w );             
                }
            }
        }
    }
    
    /* Dim 2, first half */
    
    for( a = 0 ; a < M/2 ; a += BLOCKl ){
        for( b = 0 ; b < N ; b += BLOCKc ){
            
            for( j = 0 ; j < BLOCKl ; j++ ) {
                for( i = 0 ; i < BLOCKc ; i+=4 ){
                    
                    a1 = _mm256_loadu_pd(  &W[ 2* (a+j)*lda + (b+i)] );
                    a2 =  _mm256_loadu_pd(  &W[ ( 2 * (a+j) + 1 ) *lda + (b+i)] );
                    
                    w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, deux ) );
                    _mm256_storeu_pd( &B[ j*ldb + (b+i) ], w ); 
               }
            }
        }
    }

    /* Dim 1, second half */
    
    for( a = 0 ; a < M ; a += BLOCKl ){        
        for( b = 0 ; b < N/2 ; b += BLOCKc ){
            
            for( j = 0 ; j < BLOCKl ; j++ ) {
                for( i = 0 ; i < BLOCKc ; i+=4 ){
                    
                    a1 = _mm256_set_pd( A[(a+j)*lda + 2*(b+i) + 6], A[(a+j)*lda + 2*(b+i) + 4], A[(a+j)*lda + 2*(b+i) + 2], A[(a+j)*lda + 2*(b+i)] );
                    a2 = _mm256_set_pd( A[(a+j)*lda + 2*(b+i) + 7], A[(a+j)*lda + 2*(b+i) + 5], A[(a+j)*lda + 2*(b+i) + 3], A[(a+j)*lda + 2*(b+i) + 1] );
                    
                    w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, moinsdeux ) );
                    _mm256_storeu_pd( &W[ (a+j)*ldb + (b+i) + N/2], w ); 
                }
            }
        }
    }
     
     /* Dim 2, second half */
    
    for( a = 0 ; a < M/2 ; a += BLOCKl ){
        for( b = 0 ; b < N ; b += BLOCKc ){
            
            for( j = 0 ; j < BLOCKl ; j++ ) {
                for( i = 0 ; i < BLOCKc ; i+=4 ){
                    
                    a1 = _mm256_loadu_pd(  &W[ 2* (a+j)*lda + i] );
                    a2 =  _mm256_loadu_pd(  &W[ ( 2 * (a+j) + 1 ) *lda + (b+i)] );
                    
                    w =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, moinsdeux ) );
                    _mm256_storeu_pd( &B[ ((a+j)+M/2)*ldb + (b+i) ], w ); 
                }
            }
        }
    }
}

/* Reuse the intermediate work AVX registers rather than storing in the W matrix. */
 
void dhamt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w1, w2, w2m, w1m, w;
    __m256d a1, a2, a3, a4;
    const __m256d deux = _mm256_set1_pd( 0.5 );

    /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){
            
            a1 = _mm256_set_pd( A[(j*2)*lda + 2*i + 6], A[(j*2)*lda + 2*i + 4], A[(j*2)*lda + 2*i + 2], A[(j*2)*lda + 2*i] );
            a2 = _mm256_set_pd( A[(j*2)*lda + 2*i + 7], A[(j*2)*lda + 2*i + 5], A[(j*2)*lda + 2*i + 3], A[(j*2)*lda + 2*i + 1] );
            
            a3 = _mm256_set_pd( A[(j*2+1)*lda + 2*i + 6], A[(j*2+1)*lda + 2*i + 4], A[(j*2+1)*lda + 2*i + 2], A[(j*2+1)*lda + 2*i] );
            a4 = _mm256_set_pd( A[(j*2+1)*lda + 2*i + 7], A[(j*2+1)*lda + 2*i + 5], A[(j*2+1)*lda + 2*i + 3], A[(j*2+1)*lda + 2*i + 1] );
            
            /* lines: W1 = A[i][j] + A[i+1][j] = A1 + A2 */
            
            w1 =_mm256_fmadd_pd( a1, deux, _mm256_mul_pd( a2, deux ) );
            w1m =_mm256_fmsub_pd( a1, deux, _mm256_mul_pd( a2, deux ) );

             /* lines: W2 = A[i][j+1] + A[i+1][j+1] = A3 + A4 */

            w2 =_mm256_fmadd_pd( a3, deux, _mm256_mul_pd( a4, deux ) );
            w2m =_mm256_fmsub_pd( a3, deux, _mm256_mul_pd( a4, deux ) );

            /* columns: W = W1 + W2 = W[i][j] + W[j][j+1] */
            
            w =_mm256_fmadd_pd( w1, deux, _mm256_mul_pd( w2, deux ) );
            _mm256_storeu_pd( &B[ j*ldb + i ], w ); 

            w =_mm256_fmadd_pd( w1m, deux, _mm256_mul_pd( w2m, deux ) );
            _mm256_storeu_pd( &B[ j*ldb + i + N/2], w );

            w =_mm256_fmsub_pd( w1, deux, _mm256_mul_pd( w2, deux ) );
            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i ], w ); 

            w =_mm256_fmsub_pd( w1m, deux, _mm256_mul_pd( w2m, deux ) );
            _mm256_storeu_pd( &B[ (j+M/2)*ldb + i + N/2], w ); 
        }
    }        
}

/* Reuse the intermediate work AVX registers rather than storing in the W matrix. */
 
void dhamt2_fma512_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m512d w1, w2, w2m, w1m, w;
    __m512d a1, a2, a3, a4;
    const __m512d deux = _mm512_set1_pd( 0.5 );

    /* TODO Gerer le probleme d'alignement de W pour remplacer le storeu par un store */
    
    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=8 ){
            
	  a1 = _mm512_set_pd( A[(j*2)*lda + 2*i + 14], A[(j*2)*lda + 2*i + 12], A[(j*2)*lda + 2*i + 10], A[(j*2)*lda + 2*i + 8], A[(j*2)*lda + 2*i + 6], A[(j*2)*lda + 2*i + 4], A[(j*2)*lda + 2*i + 2], A[(j*2)*lda + 2*i] );
	  a2 = _mm512_set_pd( A[(j*2)*lda + 2*i + 14], A[(j*2)*lda + 2*i + 13], A[(j*2)*lda + 2*i + 11], A[(j*2)*lda + 2*i + 9], A[(j*2)*lda + 2*i + 7], A[(j*2)*lda + 2*i + 5], A[(j*2)*lda + 2*i + 3], A[(j*2)*lda + 2*i + 1] );
	  
	  a3 = _mm512_set_pd(  A[(j*2+1)*lda + 2*i + 14], A[(j*2+1)*lda + 2*i + 12], A[(j*2+1)*lda + 2*i + 10], A[(j*2+1)*lda + 2*i + 8] , A[(j*2+1)*lda + 2*i + 6], A[(j*2+1)*lda + 2*i + 4], A[(j*2+1)*lda + 2*i + 2], A[(j*2+1)*lda + 2*i] );
	  a4 = _mm512_set_pd(  A[(j*2+1)*lda + 2*i + 15], A[(j*2+1)*lda + 2*i + 13], A[(j*2+1)*lda + 2*i + 11], A[(j*2+1)*lda + 2*i + 9], A[(j*2+1)*lda + 2*i + 7], A[(j*2+1)*lda + 2*i + 5], A[(j*2+1)*lda + 2*i + 3], A[(j*2+1)*lda + 2*i + 1] );
            
            /* lines: W1 = A[i][j] + A[i+1][j] = A1 + A2 */
            
            w1 =_mm512_fmadd_pd( a1, deux, _mm512_mul_pd( a2, deux ) );
            w1m =_mm512_fmsub_pd( a1, deux, _mm512_mul_pd( a2, deux ) );

             /* lines: W2 = A[i][j+1] + A[i+1][j+1] = A3 + A4 */

            w2 =_mm512_fmadd_pd( a3, deux, _mm512_mul_pd( a4, deux ) );
            w2m =_mm512_fmsub_pd( a3, deux, _mm512_mul_pd( a4, deux ) );

            /* columns: W = W1 + W2 = W[i][j] + W[j][j+1] */
            
            w =_mm512_fmadd_pd( w1, deux, _mm512_mul_pd( w2, deux ) );
            _mm512_storeu_pd( &B[ j*ldb + i ], w ); 

            w =_mm512_fmadd_pd( w1m, deux, _mm512_mul_pd( w2m, deux ) );
            _mm512_storeu_pd( &B[ j*ldb + i + N/2], w );

            w =_mm512_fmsub_pd( w1, deux, _mm512_mul_pd( w2, deux ) );
            _mm512_storeu_pd( &B[ (j+M/2)*ldb + i ], w ); 

            w =_mm512_fmsub_pd( w1m, deux, _mm512_mul_pd( w2m, deux ) );
            _mm512_storeu_pd( &B[ (j+M/2)*ldb + i + N/2], w ); 
        }
    }        
}

/*
c     Compute 2D Haar inverse transform of a matrix
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
c     hi haar inverse
c     mt matrix transform
c     2 2D
*/


#ifdef  __cplusplus
void dhimt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb ) {
#else
void dhimt2_initial( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb ) {
#endif
    int j, k;

    /* dim 1 */
    
    for( k = 0 ; k < M ; k++ ) {
        for( j = 0 ; j < N / 2 ; j++ ) {
            W[ k*N + 2*j] = ( A[k*lda + j] + A[ k*lda + N / 2 + j] );
            W[ k*N + 2*j + 1] = ( A[k*lda + j] - A[ k*lda + N / 2 + j] );
        }
    }
    
    /* dim 2 */
    
    for( k = 0 ; k < M / 2 ; k++ ) {
        for( j = 0 ; j < N ; j++ ) {
            B[ 2*k*ldb + j] = ( W[k*N + j] + W[ (k + M/2)*N + j] );
            B[ (2*k+1)*ldb + j] = ( W[k*N + j] - W[ (k + M/2)*N + j] );
        }
    }
    
}
 
void dhimt2_fma_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w, a1, a2;
    const __m256d moinsun = _mm256_set1_pd( -1 );
    __m256i stride =   _mm256_set_epi64x( 1*sizeof( double ), 1*sizeof( double ),0 , 0 );
    const __m256d sign = _mm256_set_pd( -1, 1, -1, 1 );

    /* dim 1 */

    /*  _mm256_i64scatter_pd would be useful to implement this one, 
        but part of AVX512 */
    
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ){
            a1 = _mm256_i64gather_pd( &A[j*lda + i], stride, 1 );
            a2 = _mm256_i64gather_pd( &A[j*lda + i + N/2], stride, 1 );
            
            w =_mm256_fmadd_pd( a2, sign, a1 );
            _mm256_storeu_pd( &W[ j*N + 2*i ], w );
            
        }
    }
    
    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i+=4 ){
            a1 = _mm256_loadu_pd(  &W[ j*N + i] );
            a2 =  _mm256_loadu_pd(  &W[ ( j + M/2 ) * N + i ] );

            w = _mm256_add_pd( a1, a2 );
            _mm256_storeu_pd( &B[ 2*j*N + i ], w ); 
            
            w =_mm256_fmadd_pd( a2, moinsun, a1 );
            _mm256_storeu_pd( &B[ (2*j+1)*N + i ], w ); 
        }
    }
    
}

void dhimt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w, a1, a2;
    const __m256d sign = _mm256_set_pd( -1, 1, -1, 1 );
    const __m256d moinsun = _mm256_set1_pd( -1 );

    /* dim 1 */

    /*  _mm256_i64scatter_pd would be useful to implement this one, 
        but part of AVX512 */
    
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ){
            a1 = _mm256_set_pd (A[j*lda + i + 1], A[j*lda + i + 1], A[j*lda + i], A[j*lda + i] );
            a2 = _mm256_set_pd( A[j*lda + i + 1 + N/2], A[j*lda + i + 1 + N/2], A[j*lda + i + N/2], A[j*lda + i + N/2] );
         
            w =_mm256_fmadd_pd( a2, sign, a1 );
            _mm256_storeu_pd( &W[ j*N + 2*i ], w );
            
        }
    }
    
    /* dim 2 */

    for( j = 0 ; j < M / 2 ; j++ ){ 
        for( i = 0 ; i < N ; i+=4 ){
            a1 = _mm256_loadu_pd(  &W[ j*N + i] );
            a2 =  _mm256_loadu_pd(  &W[ ( j + M/2 ) * N + i ] );

            w = _mm256_add_pd( a1, a2 );
            _mm256_storeu_pd( &B[ 2*j*N + i ], w ); 
            
            w =_mm256_fmadd_pd( a2, moinsun, a1 );
            _mm256_storeu_pd( &B[ (2*j+1)*N + i ], w ); 
        }
    }
    
}

void dhimt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m256d w1, w2, w1m, w2m, a1, a2, a3, a4;
    const __m256d sign = _mm256_set_pd( -1, 1, -1, 1 );
    const __m256d moinsun = _mm256_set1_pd( -1 );

    /* dim 1 */

    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=2 ){ 
            a1 = _mm256_set_pd (A[j*lda + i + 1], A[j*lda + i + 1], A[j*lda + i], A[j*lda + i] );
            a2 = _mm256_set_pd( A[j*lda + i + 1 + N/2], A[j*lda + i + 1 + N/2], A[j*lda + i + N/2], A[j*lda + i + N/2] );

            a3 = _mm256_set_pd (A[( j + M/2 )*lda + i + 1], A[( j + M/2 )*lda + i + 1], A[( j + M/2 )*lda + i], A[( j + M/2 )*lda + i] );
            a4 = _mm256_set_pd( A[( j + M/2 )*lda + i + 1 + N/2], A[( j + M/2 )*lda + i + 1 + N/2], A[( j + M/2 )*lda + i + N/2], A[( j + M/2 )*lda + i + N/2] );

            /* Dim 1 */
            
            w1  = _mm256_fmadd_pd( a2, sign, a1 );
            w1m = _mm256_fmadd_pd( a4, sign, a3 );

            /* Dim 2 */

            w2  = _mm256_add_pd( w1, w1m );
            w2m = _mm256_fmadd_pd( w1m, moinsun, w1 );

            /* Store */
            
            _mm256_storeu_pd( &B[ 2*j*N + 2*i ], w2 );
            _mm256_storeu_pd( &B[ (2*j+1)*N + 2*i ], w2m );

        }
    }
    
}

 void dhimt2_fma512_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb ) {
    int i, j;
    __m512d w1, w2, w1m, w2m, a1, a2, a3, a4;
    const __m512d sign = _mm512_set_pd( -1, 1, -1, 1, -1, 1, -1, 1 );
    const __m512d moinsun = _mm512_set1_pd( -1 );

    /* dim 1 */

    for( j = 0 ; j < M/2 ; j++ ) {
        for( i = 0 ; i < N / 2 ; i+=4 ){ 
            a1 = _mm512_set_pd( A[j*lda + i + 3], A[j*lda + i + 3], A[j*lda + i + 2], A[j*lda + i + 2], A[j*lda + i + 1], A[j*lda + i + 1], A[j*lda + i], A[j*lda + i] );
            a2 = _mm512_set_pd( A[j*lda + i + 3 + N/2], A[j*lda + i + 3 + N/2], A[j*lda + i + 2 + N/2], A[j*lda + i + 2 + N/2], A[j*lda + i + 1 + N/2], A[j*lda + i + 1 + N/2], A[j*lda + i + N/2], A[j*lda + i + N/2] );

            a3 = _mm512_set_pd( A[( j + M/2 )*lda + i + 3], A[( j + M/2 )*lda + i + 3], A[( j + M/2 )*lda + i + 2], A[( j + M/2 )*lda + i + 2], A[( j + M/2 )*lda + i + 1], A[( j + M/2 )*lda + i + 1], A[( j + M/2 )*lda + i], A[( j + M/2 )*lda + i] );
            a4 = _mm512_set_pd( A[( j + M/2 )*lda + i + 3 + N/2], A[( j + M/2 )*lda + i + 3 + N/2], A[( j + M/2 )*lda + i + 2 + N/2], A[( j + M/2 )*lda + i + 2 + N/2], A[( j + M/2 )*lda + i + 1 + N/2], A[( j + M/2 )*lda + i + 1 + N/2], A[( j + M/2 )*lda + i + N/2], A[( j + M/2 )*lda + i + N/2] );

            /* Dim 1 */
            
            w1  = _mm512_fmadd_pd( a2, sign, a1 );
            w1m = _mm512_fmadd_pd( a4, sign, a3 );

            /* Dim 2 */

            w2  = _mm512_add_pd( w1, w1m );
            w2m = _mm512_fmadd_pd( w1m, moinsun, w1 );

            /* Store */
            
            _mm512_storeu_pd( &B[ 2*j*N + 2*i ], w2 );
            _mm512_storeu_pd( &B[ (2*j+1)*N + 2*i ], w2m );

        }
    }
    
}

/*
c     Compute 1D Haar direct transform of a matrix
c
c     Params:
c     A: input matrix M*N, double precision
c     B: output matrix M*N, double precision
c     N: nb of columns of the matrix, integer

c     TODO ASSUME M AND N ARE POWERS OF TWO
c     TRANS = 'T': column-major, 'N': row-major

c     Name:
c     d double
c     ha haar
c     mt matrix transform
*/

void dhamt_loop( double* restrict A, double* restrict B, int N ) {
    int i;
    
    for( i = 0 ; i < N / 2 ; i++ ){ 
        B[ i] = ( A[2*i] + A[2*i+1] ) / 2.0;
        }
    for( i = 0 ; i < N / 2 ; i++ ){ 
        B[ i + N/2] = ( A[2*i] - A[ 2*i+1] ) / 2.0;
    }    
}

/*
c     Compute 1D Haar inverse transform of a matrix
c
c     Params:
c     A: input matrix M*N, double precision
c     B: output matrix M*N, double precision
c     N: nb of columns of the matrix, integer

c     TODO ASSUME M AND N ARE POWERS OF TWO
c     TRANS = 'T': column-major, 'N': row-major

c     Name:
c     d double
c     hi haar inverse
c     mt matrix transform
*/

void dhimt_loop( double* restrict A, double* restrict B, int N ) {
    int i;
    
    for( i = 0 ; i < N / 2 ; i++ ) {
        B[ 2*i] = ( A[i] + A[ N / 2 + i] );
        B[2*i + 1] = ( A[i] - A[N / 2 + i] );
    }    
}

