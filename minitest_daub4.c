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

#include "matrices.h"

/* If PAPI does not work (indicates 0 counters):
   echo 0 > /proc/sys/kernel/perf_event_paranoid
*/

#define NUMRUN 1

#define DEFAULTM 16
#define DEFAULTN 16


void DDA4MT( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void DDI4MT( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );


#ifndef WITHPAPI
extern __inline__ long long rdtsc(void) {
  long long a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}
#endif



int main( int argc, char** argv ){

    int M, N;
    double* mat;
    double* work;
    double* wor2;
    double cond = 1e10;

#ifdef WITHPAPI
    #define NUM_EVENTS 3
    int i, rc;
    //    int events[NUM_EVENTS] = { PAPI_TOT_CYC, PAPI_L2_DCM, PAPI_L3_TCM };
    int events[NUM_EVENTS] = { PAPI_TOT_CYC, PAPI_L2_LDM, PAPI_L2_STM };
    long long values[NUM_EVENTS] = {0, 0, 0};
    int num_hwcntrs = 0;
    if ((num_hwcntrs = PAPI_num_counters()) <= PAPI_OK)  {
        printf( "Not enough counters: %d available\n", num_hwcntrs );
        return EXIT_FAILURE;
    }
#else
    long long t_start, t_end;
    int i;
#endif
    
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
    memset( wor2, 0, M*N*sizeof( double ) );

    /* Ill-conditionned matrix */
    
    //    dillrandom( mat, M, N, N, cond, work, wor2 );
        drandom( mat, M, N );
    //identity( mat, M, N );
        printmatrixOctave( mat, M, N );
        //  printmatrixOctave( basarab, M, N );
    //printmatrix( mat, M, N );
    
#ifdef WITHPAPI
        rc = PAPI_start_counters( events, NUM_EVENTS ); 
        if( rc  != PAPI_OK ){
            PAPI_perror( "Error starting the counters" );
        }
#else
        t_start = rdtsc();
#endif
    /* Daub4 transform */
    for( i = 0 ; i < NUMRUN ; i++ ) {
        DDA4MT( mat, work, wor2, M, N, N, N );
    }
    //   printmatrixOctave( wor2, M, N );
    //    printmatrixOctave( work, M, N );
    //         printmatrix( work, M, N );
    
    memset( wor2, 0, M*N*sizeof( double ) );
    memset( mat, 0, M*N*sizeof( double ) );
    DDI4MT( work, mat, wor2, M, N, N, N );
    //    printmatrixOctave( wor2, M, N );
    printmatrixOctave( mat, M, N );
    
#ifdef WITHPAPI
    PAPI_stop_counters( values, NUM_EVENTS );
    //    printf( "# M \t N \t PAPI_TOT_CYC \t PAPI_L2_DCM \t PAPI_L3_TCM \t PAPI_L1_LDM \t PAPI_L1_STM\n" );
    printf( "# M \t N \t PAPI_TOT_CYC \t PAPI_L1_LDM \t PAPI_L1_STM\n" );
    printf( "%d \t %d \t %lld \t %lld \t %lld \n", M, N, values[0] / NUMRUN, values[1] / NUMRUN , values[2] / NUMRUN );
#else
    t_end = rdtsc();
    printf( "# M \t N \t time \n" );
    printf( "%d \t %d \t %lld \n", M, N, (t_end - t_start ) / NUMRUN );

#endif

    free( mat );
    free( work );
    free( wor2 );
    return EXIT_SUCCESS;
}

