/* This file is part of pawt.
 *
 * Calling program for the 2D Haar transform dhamt2.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WITHPAPI
#include <papi.h>
#endif

#include "../matrices.h"
#include "utils.h"

#define DEFAULTM 1024
#define DEFAULTN 1024

void dhamt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dhamt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dhamt2_avx( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_avx_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_sse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_fma_block( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );


int main( int argc, char** argv ){

    int M, N;
    double* mat;
    double* work;
    double* wor2;
    double* reference;
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
    reference = (double*) malloc( M*N*sizeof( double));
    memset( wor2, 0, M*N*sizeof( double ) );

    /* Ill-conditionned matrix */
    
    //    dillrandom( mat, M, N, N, cond, work, wor2 );
    drandom( mat, M, N );

    /* Initial: point of reference */
    
    dhamt2_initial( mat, reference, wor2, M, N, N, N );    

    dhamt2_loop( mat, work, wor2, M, N, N, N );    
    if( compareMatrices( reference, work, M, N ) ) {
        fprintf( stderr, "dhamt2_loop: OK\n" );
    } else {
        fprintf( stderr, "dhamt2_loop does not match the reference\n" );
    }

    dhamt2_avx( mat, work, wor2, M, N, N, N );    
    if( compareMatrices( reference, work, M, N ) ) {
        fprintf( stderr, "dhamt2_avx: OK\n" );
    } else {
        fprintf( stderr, "dhamt2_avx does not match the reference\n" );
    }

    dhamt2_avx_gather( mat, work, wor2, M, N, N, N );    
    if( compareMatrices( reference, work, M, N ) ) {
        fprintf( stderr, "dhamt2_avx_gather: OK\n" );
    } else {
        fprintf( stderr, "dhamt2_avx_gather does not match the reference\n" );
    }

    dhamt2_sse( mat, work, wor2, M, N, N, N );    
    if( compareMatrices( reference, work, M, N ) ) {
        fprintf( stderr, "dhamt2_sse: OK\n" );
    } else {
        fprintf( stderr, "dhamt2_sse does not match the reference\n" );
    }

    dhamt2_fma( mat, work, wor2, M, N, N, N );    
    if( compareMatrices( reference, work, M, N ) ) {
        fprintf( stderr, "dhamt2_fma: OK\n" );
    } else {
        fprintf( stderr, "dhamt2_fma does not match the reference\n" );
    }

    dhamt2_fma_reuse( mat, work, wor2, M, N, N, N );    
    if( compareMatrices( reference, work, M, N ) ) {
        fprintf( stderr, "dhamt2_fma_reuse: OK\n" );
    } else {
        fprintf( stderr, "dhamt2_fma_reuse does not match the reference\n" );
    }

    free( mat );
    free( work );
    free( wor2 );
    free( reference );
    return EXIT_SUCCESS;
}

