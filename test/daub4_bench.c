/* This file is part of pawt.
 *
 * Benchmarking program for the 2D Daubechies D4 transform dda4mt2.
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
#include <execinfo.h>

#include "../matrices.h"
#include "utils.h"
#include "benchmarking.h"
#include "daub4_all.h"

#define DEFAULTN 1024
#define NUMRUN 25
#define MINSIZE 16

/* If PAPI does not work (indicates 0 counters):
   sudo bash -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
*/

typedef void (*functionvariant_t)( double*, double*, double*, int, int, int, int );
#define FUNC_DEF(func) { func, #func },
typedef struct {
    functionvariant_t func;
    const char * name;
} implem_t;

int main( int argc, char** argv ){

    int N, i, j, s, previous;
    double* mat;
    double* work;
    double* wor2;
    //    double cond = 1e10;

    implem_t direct[] = {
        FUNC_DEF( dda4mt2_initial )
        FUNC_DEF( dda4mt2_loop )
#if defined( __SSE__ ) || defined( __aarch64__ )
        FUNC_DEF( dda4mt2_sse )
        FUNC_DEF( dda4mt2_sse_peel )
        FUNC_DEF( dda4mt2_sse_reuse )
        FUNC_DEF( dda4mt2_sse_reuse_peel )
#endif //  __SSE__ ||  __aarch64__
#ifdef __AVX__
        FUNC_DEF( dda4mt2_avx )
        FUNC_DEF( dda4mt2_avx_peel )
        FUNC_DEF( dda4mt2_avx_gather )
#endif // __AVX__
#ifdef __AVX2__
        FUNC_DEF( dda4mt2_fma )
        FUNC_DEF( dda4mt2_fma2)
        FUNC_DEF( dda4mt2_fma_reuse )
        FUNC_DEF( dda4mt2_fma_reuse_peel )
        FUNC_DEF( dda4mt2_fma2_reuse )
        FUNC_DEF( dda4mt2_fma2_reuse_gather )
        FUNC_DEF( dda4mt2_fma2_reuse_gather_peel )
#endif // __AVX2__
#ifdef __AVX512F__
        FUNC_DEF( dda4mt2_fma512_reuse )
        FUNC_DEF( dda4mt2_fma512_reuse_gather )
#endif // __AVX512F__
       NULL
    };
    implem_t backward[] = {
        FUNC_DEF( ddi4mt2_loop )
#if defined( __SSE__ ) || defined( __aarch64__ )
        FUNC_DEF( ddi4mt2_sse )
        FUNC_DEF( ddi4mt2_sse_reuse )
        FUNC_DEF( ddi4mt2_sse_peel )
        FUNC_DEF( ddi4mt2_sse_reuse_peel )
#endif //  __SSE__ ||  __aarch64__
#ifdef __AVX__
        FUNC_DEF( ddi4mt2_avx )
        FUNC_DEF( ddi4mt2_avx_peel )
        FUNC_DEF( ddi4mt2_avx_gather )
#endif // __AVX__
#ifdef __AVX2__
        FUNC_DEF( ddi4mt2_fma )
        FUNC_DEF( ddi4mt2_fma_gather )
        FUNC_DEF( ddi4mt2_fma2 )
        FUNC_DEF( ddi4mt2_fma2_gather )
        FUNC_DEF( ddi4mt2_fma_reuse )
#endif // __AVX2__
#ifdef __AVX512F__
        FUNC_DEF( ddi4mt2_fma512_reuse )
#endif // __AVX512F__
       NULL
    };

#ifdef WITHPAPI
    #define NUM_EVENTS 3
    int rc;
    int events[NUM_EVENTS] = { PAPI_TOT_CYC, PAPI_L2_DCM, PAPI_L3_TCM };
    long long values[NUM_EVENTS] = {0, 0, 0};
    int num_hwcntrs = 0;
    if ((num_hwcntrs = PAPI_num_counters()) <= PAPI_OK)  {
        printf( "Not enough counters: %d available\n", num_hwcntrs );
        return EXIT_FAILURE;
    }
#else
    unsigned long long t_start, t_end;
#endif

    if( argc < 2 ) {
        N = DEFAULTN;
    } else {
        N = atoi( argv[1] );
    }
    mat  = (double*) malloc( N*N*sizeof( double));
    work = (double*) malloc( N*N*sizeof( double));
    wor2 = (double*) malloc( N*N*sizeof( double));
    memset( wor2, 0, N*N*sizeof( double ) );

    /* Ill-conditionned matrix */

    //    dillrandom( mat, N, N, N, cond, work, wor2 );
    drandom( mat, N, N );

#ifdef WITHPAPI
    printf( "# %*s \t N \t N \t PAPI_TOT_CYC \t PAPI_L2_DCM \t PAPI_L3_TCM \t cycles per element\n", 25," " );
#else
    printf( "# %*s \t N \t N \t time \n", 25, " " );
#endif

    /* Direct transform */

    s = MINSIZE;
    previous = MINSIZE / 2;
    while( s < N ) {

        for( i = 0 ; direct[i].func != NULL ; i++ ) {
            drandom( mat, s, s ); // re-init
            STARTCOUNTERS();
            for( j = 0 ; j < NUMRUN ; j++ ) {
                direct[i].func( mat, work, wor2, s, s, s, s );
            }
            ENDCOUNTERS( direct[i].name );
        }

        if( 1 == __builtin_popcount( s ) ) {
            s += previous;
        } else {
            s += previous;
            previous *= 2;
        }
    }

    /* Backward transform */

    s = MINSIZE;
    previous = MINSIZE / 2;
    while( s < N ) {

        for( i = 0 ; backward[i].func != NULL ; i++ ) {
            drandom( work, s, s ); // re-init
            STARTCOUNTERS();
            for( j = 0 ; j < NUMRUN ; j++ ) {
                backward[i].func( work, mat, wor2, s, s, s, s );
            }
            ENDCOUNTERS( backward[i].name );
        }

        if( 1 == __builtin_popcount( s ) ) {
            s += previous;
        } else {
            s += previous;
            previous *= 2;
        }
    }

    free( mat );
    free( work );
    free( wor2 );
    return EXIT_SUCCESS;
}
