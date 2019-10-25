#ifdef __aarch64__

#define double    float64_t

// SSE instructions

#define __m128d   float64x2_t

#define _mm_add_pd( A, B )     vaddq_f64( A, B )
#define _mm_sub_pd( A, B )     vsubq_f64( A, B )
#define _mm_mul_pd( A, B )     vmulq_f64( A, B )

#define _mm_storeu_pd( p, B )  vst1q_f64( p, B )
#define _mm_store_pd( p, B )   vst1q_f64( __builtin_assume_aligned( p, 16 ), B )
#define _mm_loadu_pd( p )      vld1q_f64( p )
#define _mm_load_pd( p )       vld1q_f64( __builtin_assume_aligned( p, 16 ) )
#define _mm_set1_pd( A )       vdupq_n_f64( A )

inline float64x2_t _mm_set_pd( double A, double B ) {
	double __attribute__((aligned(16))) data[2] = { B, A };
	return vld1q_f64(data);
}

#endif
