#ifdef WITHPAPI
#include <papi.h>
#endif

#ifndef WITHPAPI
extern __inline__ long long rdtsc(void) {
  long long a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}
#endif

#ifdef WITHPAPI
#define STARTCOUNTERS() do { \
        rc = PAPI_start_counters( events, NUM_EVENTS );  \
        if( rc  != PAPI_OK ){                            \
            printf( "Error starting the counters\n" );   \
        }                                                \
    } while( 0 );
#define ENDCOUNTERS( name ) do {                                        \
        PAPI_stop_counters( values, NUM_EVENTS );                       \
        printf( "%*s \t %d \t %d \t %*lld \t %*lld \t %*lld \t %*lld\n", 25, name, s, s, 8, values[0] / NUMRUN, 8, values[1] / NUMRUN, 8 , values[2] / NUMRUN, 8, values[0] / (s * s * NUMRUN ) ); \
    } while( 0 );
#else
#define STARTCOUNTERS() do {                             \
        t_start = rdtsc();                               \
    } while( 0 );
#define ENDCOUNTERS( name ) do {                                        \
        t_end = rdtsc();                                                \
        printf( "%*s \t %d \t %d \t %lld \t %lld \n",25, name, s, s, (t_end - t_start ) / NUMRUN, (t_end - t_start ) / ( s * s * NUMRUN ) ); \
    } while( 0 );
#endif


