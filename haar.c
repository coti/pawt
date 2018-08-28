#include <string.h>

#include <x86intrin.h>

//#define dhamt2_initial dhamt2_ 
//#define dhamt2_loop dhamt2_
//#define dhamt2_avx dhamt2_ 
#define dhamt2_sse dhamt2_ 

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
            a1 = _mm256_load_pd(  &W[ 2* j*lda + i] );
            a2 =  _mm256_load_pd(  &W[ ( 2 * j + 1 ) *lda + i] );
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
