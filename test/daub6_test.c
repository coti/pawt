/* This file is part of pawt.
 *
 * Calling program for the 2D Daubechies D4 transform dda4mt2.
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

void dda6mt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dda6mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb );

typedef void (*functionvariant_t)( double*, double*, double*, int, int, int, int );
#define FUNC_DEF(func) { func, #func },
typedef struct {
    functionvariant_t func;
    const char * name;
} implem_t;

int main( int argc, char** argv ){

    int M, N, i;
    double* mat;
    double* work;
    double* wor2;
    double* reference;
    double cond = 1e10;
    
    implem_t direct[] = {
        FUNC_DEF( dda6mt2_initial )
        FUNC_DEF( dda6mt2_loop )
        NULL
    };
    implem_t backward[] = {
       NULL
    };

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
    
    i = 0;
    direct[i].func( mat, work, wor2, M, N, N, N );    
    i++;

    while( direct[i].func != NULL ) {
        direct[i].func( mat, work, wor2, M, N, N, N );    
        if( compareMatrices( reference, work, M, N ) ) {
            fprintf( stderr, "%s: OK\n", direct[i].name );
        } else {
            fprintf( stderr, "%s does not match the reference\n", direct[i].name );
        }
        i++;
    }

    /* Backward transform */
    
    i = 0;
    while( backward[i].func != NULL ) {
        backward[i].func( work, reference, wor2, M, N, N, N );    
        if( compareMatrices( reference, mat, M, N ) ) {
            fprintf( stderr, "%s: OK\n", backward[i].name );
        } else {
            fprintf( stderr, "%s does not match the reference\n", backward[i].name );
        }
        i++;
    }
    
    free( mat );
    free( work );
    free( wor2 );
    free( reference );
    return EXIT_SUCCESS;
}
