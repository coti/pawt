/* This file is part of pawt.
 *
 * Calling program for the 2D Haar transform dhamt2.
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

/* Direct Haar transforms */

void dhamt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dhamt2_loop( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dhamt2_avx( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_avx_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_sse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_fma_block( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhamt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );

/* Backward Haar transforms */

void dhimt2_initial( double* A, double* B, double* W, int M, int N, int lda, int ldb );
void dhimt2_fma_gather( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhimt2_fma( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );
void dhimt2_fma_reuse( double*  A, double*  B, double*  W, int M, int N, int lda, int ldb );