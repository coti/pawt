void dda4mt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dda4mt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb );
#if defined( __SSE__ ) || defined( __aarch64__ )
void dda4mt2_sse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_sse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_sse_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_sse_reuse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
#endif //  __SSE__ ||  __aarch64__
#ifdef __AVX__
void dda4mt2_avx( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_avx_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_avx_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
#endif // __AVX__
#ifdef __AVX2__
void dda4mt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma_reuse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2_reuse_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma2_reuse_gather_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
#endif // __AVX2__
#ifdef __AVX512F__
void dda4mt2_fma512_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dda4mt2_fma512_reuse_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
#endif // __AVX512F__

void ddi4mt2_loop( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
#if defined( __SSE__ ) || defined( __aarch64__ )
void ddi4mt2_sse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void ddi4mt2_sse_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void ddi4mt2_sse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void ddi4mt2_sse_reuse_peel( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
#endif //  __SSE__ ||  __aarch64__
#ifdef __AVX__
void ddi4mt2_avx( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_avx_peel( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_avx_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb );
#endif // __AVX__
#ifdef __AVX2__
void ddi4mt2_fma( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_fma_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void ddi4mt2_fma_reuse( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void ddi4mt2_fma2( double* restrict A, double* restrict B, double* restrict W, int M, int N, int lda, int ldb );
void ddi4mt2_fma2_gather( double* A, double* B, double* W, int M, int N, int lda, int ldb );
#endif // __AVX2__
#ifdef __AVX512F__
void ddi4mt2_fma512_reuse( double* A, double* B, double* W, int M, int N, int lda, int ldb );
#endif // __AVX512F__
