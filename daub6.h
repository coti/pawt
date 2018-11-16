/* This file is part of pawt.
 *
 * Various implementations of the 2D Daubechies D6 transform dda6mt2
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

static inline void dGetCoeffs6( double* h, double* g ) {
    int i, one;

    h[0] = (1+sqrt(10) + sqrt(5+2*sqrt(10) ) )/(16 *sqrt(2)) ;
    h[1] = (5+sqrt(10) + 3*sqrt(5+2*sqrt(10) ))/(16 *sqrt(2)) ;
    h[2] = (10-2*sqrt(10) + 2*sqrt(5+2*sqrt(10) ))/(16 *sqrt(2));
    h[3] = (10-2*sqrt(10) - 2*sqrt(5+2 *sqrt(10) ))/(16 *sqrt(2)) ;
    h[4] = (5+sqrt(10) - 3*sqrt(5+2 *sqrt(10) ))/(16 *sqrt(2));
    h[5] = (1+sqrt(10) - sqrt(5+2*sqrt(10) ))/(16 *sqrt(2)) ;

    one = 1;
    for( i = 0 ; i < 6 ; i++ ) {
        g[i] = one * h[6 - 1 - i];
        one *= -1;
    }
}
