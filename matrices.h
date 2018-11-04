void ddiago( double* A, int M, int N, double cond );
void identity( double* mat, int M, int N );
void drandom( double* A, int M, int N );
void dillrandom( double* A, int M, int N, int lda, double cond, double* work, double* work2 );
void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* ); 
void printmatrix( double* mat, int M, int N );
void printmatrixOctave( double* mat, int M, int N );
