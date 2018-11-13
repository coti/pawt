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

#ifdef DEBUG_BASARAB
    double basarab[64] =  {0.1206, 0.6438, 0.0623, 0.4903, 0.3061, 0.8164, 0.9972, 0.4246 , 0.7675, 0.8468, 0.1681, 0.4045, 0.3025, 0.7730, 0.3156, 0.8355, 0.3103, 0.4922, 0.0378, 0.6989, 0.1704, 0.4167, 0.1199, 0.2274 , 0.3527, 0.1086, 0.8734, 0.9629, 0.5332, 0.4056, 0.8503, 0.1604 , 0.7382, 0.8833, 0.3093, 0.4463, 0.0403, 0.9273, 0.7538, 0.5861, 0.7008, 0.9463, 0.4652, 0.3890, 0.4388, 0.3014, 0.8448, 0.0802, 0.6985, 0.4762, 0.1508, 0.5055, 0.8133, 0.1878, 0.3805, 0.5890, 0.6836, 0.0463, 0.0702, 0.4994, 0.2996, 0.6929, 0.0510, 0.6763  };

    double solution[64] = { 1.219168 ,   0.482832 ,   1.111272  ,  1.119340  ,  -0.581676   , -0.357569  ,  -0.187398  ,  -0.228104     ,
                          0.604543  ,  1.379706  ,  0.964038  ,  0.769794 ,   -0.099787  ,  -0.306024  ,  0.295851 ,   0.050487     ,
                          1.641438 ,   0.799625   , 1.008289,    0.941142   , -0.240047 ,   -0.076741  ,  0.556083 ,   -0.028614  ,   
                          0.592649 ,   0.714927  ,  0.938438 ,   1.046255  ,  -0.420245  ,  0.021793  ,  -0.318054  ,  0.312924     ,
                          -0.068692 ,   -0.272928   , -0.310585,    -0.387884,    -0.354573,    -0.060916 ,   -0.307149 ,   -0.028895,     
                          0.217629 ,   -0.315991   , 0.205230 ,   0.341722 ,   -0.145144 ,   -0.453236 ,   -0.240130   , -0.048585    , 
                          0.025795  ,  0.149668   , -0.104576  ,  0.057637 ,   -0.072589  ,  0.536268 ,   -0.003323   , -0.039721     ,
                          -0.219055  ,  -0.030570,    0.253523  ,  -0.053564,    -0.018347 ,   -0.115078 ,   0.631844  ,  -0.475752    }; 
#endif
    
#ifdef WITHPAPI
    #define NUM_EVENTS 3
    int i, rc;
    int events[NUM_EVENTS] = { PAPI_TOT_CYC, PAPI_L2_DCM, PAPI_L3_TCM };
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
#ifndef DEBUG_BASARAB
    mat  = (double*) malloc( M*N*sizeof( double));
#else
    mat = basarab;
#endif
    work = (double*) malloc( M*N*sizeof( double));
    wor2 = (double*) malloc( M*N*sizeof( double));
    memset( wor2, 0, M*N*sizeof( double ) );

    /* Ill-conditionned matrix */
    
    //    dillrandom( mat, M, N, N, cond, work, wor2 );
        drandom( mat, M, N );
    //identity( mat, M, N );
        //    printmatrixOctave( mat, M, N );
        //  printmatrixOctave( basarab, M, N );
    //printmatrix( mat, M, N );
    
#ifdef WITHPAPI
        rc = PAPI_start_counters( events, NUM_EVENTS ); 
        if( rc  != PAPI_OK ){
            printf( "Error starting the counters\n" );
        }
#else
        t_start = rdtsc();
#endif
    /* Daub4 transform */
    for( i = 0 ; i < NUMRUN ; i++ ) {
        DDA4MT( mat, work, wor2, M, N, N, N );
        //    printmatrixOctave( work, M, N );
        //         printmatrix( work, M, N );
    }
    /*    printmatrixOctave( wor2, M, N );
    printmatrixOctave( work, M, N );
    printmatrixOctave( solution, M, N );
    */
#ifdef WITHPAPI
    PAPI_stop_counters( values, NUM_EVENTS );
    printf( "# M \t N \t PAPI_TOT_CYC \t PAPI_L2_DCM \t PAPI_L3_TCM\n" );
    printf( "%d \t %d \t %lld \t %lld \t %lld\n", M, N, values[0] / NUMRUN, values[1] / NUMRUN , values[2] / NUMRUN );
#else
    t_end = rdtsc();
    printf( "# M \t N \t time \n" );
    printf( "%d \t %d \t %lld \n", M, N, (t_end - t_start ) / NUMRUN );

#endif

#ifndef DEBUG_BASARAB
    free( mat );
#endif
    free( work );
    free( wor2 );
    return EXIT_SUCCESS;
}

