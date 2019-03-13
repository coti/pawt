/* This file is part of pawt.
 *
 * Various implementations of the 2D Daubechies D8 transform dda4mt2
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

static inline void dGetCoeffs8( double* c, double* d  ) {
    int i, one;

    c[0] = 0.32580343;
    c[1] = 1.01094572;
    c[2] = 0.89220014;
    c[3] = -0.03957503;
    c[4] = -0.26450717;
    c[5] = 0.0436163;
    c[6] = 0.0465036;
    c[7] = -0.01498699;

    one = 1;
    for( i = 0 ; i < 8 ; i++ ) {
        d[i] = one * c[8 - 1 - i];
        one *= -1;
    }
}
