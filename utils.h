inline void print256( __m256d r ) {
    double d[4];
    int i;
    _mm256_storeu_pd( d, r );
    for( i = 0 ; i < 4 ; i++ ) printf( "%.4f  ", d[i] );
    printf( "\n" );
}
