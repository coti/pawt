#include <stdbool.h>

#define PRECISION 1e-8

static inline bool compare( double d1, double d2 ) {
    if((  ( d2 - PRECISION < d1 ) && ( d1 < d2 + PRECISION ) )
       || ( ( d1 - PRECISION < d2 ) && ( d2 < d1 + PRECISION ) )) {
        return true;
    }
    return false;
} 

static inline bool compareMatrices( double* m1, double* m2, int M, int N ) {
    int i, j;
    for( i = 0 ; i < M ; i++ ) {
        for( j = 0 ; j < N ; j++ ) {
            if( ! compare( m1[ i * N + j], m1[ i * N + j] ) ) {
                fprintf( stderr, "Numbers at line %d, col %d, do not match\n", i, j );
                return false;
            }
        }
    }
    return true;
}
