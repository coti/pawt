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
    int i, j;

    dGetCoeffs4( &h0, &h1, &h2, &h3 );
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;


    /* dim 1 */
    for( j = 0 ; j < M ; j++ ) {
        for( i = 0 ; i < (N / 2)-1 ; i++ ){ 
            W[ j*N + i] =  h0 * A[j*lda + 2*i] + h1 * A[ j*lda + 2*i + 1] 
                + h2 * A[j*lda + 2*i +2] + h3 * A[ j*lda + 2*i + 3];
        }
        W[ j*N + N/2 - 1] =  h0 * A[j*lda + N - 2] + h1 * A[ j*lda + N - 1]; // last cols

        for( i = 0 ; i < (N / 2)-1 ; i++ ){ 
            W[ j*N + i + N/2] =  g0 * A[j*lda + 2*i] + g1 * A[ j*lda + 2*i + 1] 
                + g2 * A[j*lda + 2*i + 2] + g3 * A[ j*lda + 2*i + 3];
       }
        W[ j*N + N - 1] =  g0 * A[j*lda + N - 2] + g1 * A[ j*lda + N - 1]; // last cols
   }
    
    /* dim 2 */
    
    for( j = 0 ; j < (M / 2)-1 ; j++ ){ 
        for( i = 0 ; i < N ; i++ ){
            B[i + j * ldb] = h0 * W[2*j*N +i] + h1 * W[ (2*j+1)*N + i] 
                + h2 * W[(2*j+2)*N + i] + h3 * W[ (2*j+3)*N + i];
            B[i + (j+M/2) * ldb ] = g0 * W[2*j*N +i] + h1 * W[ (2*j+1)*N + i] 
                + h2 * W[(2*j+2)*N + i] + h3 * W[ (2*j+3)*N + i];
            // printf( "B[%d] = %.02lf = %.02lf +%.02lf +%.02lf +%.02lf \n", i + j * ldb, B[i + j * ldb], h0 * W[2*j*N +i], h1 * W[ (2*j+1)*N + i], h2 * W[(2*j+2)*N + i], h3 * W[ (2*j+3)*N + i] );
            //  printf( "B[%d] = %.02lf = %.02lf +%.02lf +%.02lf +%.02lf \n", i + (j + M/2) * ldb, B[i + (j+M/2) * ldb], g0 * W[2*j*N +i], h1 * W[ (2*j+1)*N + i], h2 * W[(2*j+2)*N + i], h3 * W[ (2*j+3)*N + i] );
      }
    }
    for( i = 0 ; i < N ; i++ ){ // last lines
        B[i + ((M-1)/2) * ldb] = h0 * W[(M-2)*N +i] + h1 * W[ (M-1)*N + i];
     
        B[i + (M-1) * ldb ] = g0 * W[(M-2)*N +i] + g1 * W[ (M-1)*N + i] ;
        //        printf( "B[%d] = %.02lf = %.02lf +%.02lf \n", i + ((M-1)/2) * ldb, B[i + ((M-1)/2) * ldb],  h0 * W[(M-2)*N +i], h1 * W[ (M-1)*N + i] );
        //        printf( "B[%d] = %.02lf = %.02lf +%.02lf \n", i + (j + M/2) * ldb, B[i + (M-1) * ldb ], g0 * W[(M-2)*N +i], g1 * W[ (M-1)*N + i] );
    }
    

}

