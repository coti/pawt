/* This file is part of pawt.
 *
 * Various implementations of the 2D Daubechies D4 transform dda4mt2
 * (headers file).
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

static inline void dGetCoeffs4( double* c0, double* c1, double* c2, double* c3  ) {
    *c0 = (1.0 + sqrt( 3 ) ) / ( 4.0 * sqrt( 2 ) );
    *c1 = (3.0 + sqrt( 3 ) ) / ( 4.0 * sqrt( 2 ) );
    *c2 = (3.0 - sqrt( 3 ) ) / ( 4.0 * sqrt( 2 ) );
    *c3 = (1.0 - sqrt( 3 ) ) / ( 4.0 * sqrt( 2 ) );
}
