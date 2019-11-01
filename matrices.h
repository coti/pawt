void ddiago( double* A, int M, int N, double cond );
void identity( double* mat, int M, int N );
void drandom( double* A, int M, int N );
void dillrandom( double* A, int M, int N, int lda, double cond, double* work, double* work2 );
void printmatrix( double* mat, int M, int N );
void printmatrixOctave( double* mat, int M, int N );

#ifdef __aarch64__
#include "armpl.h"
 void dgemm_(const char *transa, const char *transb, const armpl_int_t *m, const armpl_int_t *n, const armpl_int_t *k, const double *alpha, const double *a, const armpl_int_t *lda, const double *b, const armpl_int_t *ldb, const double *beta, double *c, const armpl_int_t *ldc, ... );
#else
void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* ); 
#endif // __aarch64__
