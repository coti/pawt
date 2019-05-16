void dda4mt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dda4mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dda4mt2_avx( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_avx_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2_reuse_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );

void ddi4mt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_avx( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_avx_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void ddi4mt2_fma( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_fma_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void ddi4mt2_fma_reuse( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void ddi4mt2_fma2( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_fma2_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb );