## Possible DHAMT values:
# dhamt2_initial: initial, naive
# dhamt2_loop: loop order optimization
# dhamt2_avx: avx, using set_pd to load the data
# dhamt2_avx_gather: avx, using gather_pd
# dhamt2_sse: sse implementation for older process (pre-Haswell)
# dhamt2_fma: fused multiply-and-add implementation
# dhamt2_fma_reuse: fused multiply-and-add implementation using indermediate AVX registers rather than W matrix

DHAMT=dhamt2_fma_reuse

# dda4mt2_initial
DDA4MT=dda4mt2_avx

# This file is part of pawt.
#
# Makefile, to compile the program
#
# Copyright 2018 LIPN, CNRS UMR 7030, Université Paris 13, 
#                Sorbonne-Paris-Cité. All rights reserved.
# Author: see AUTHORS
# Licence: GPL-3.0, see COPYING for details.
#
# Additional copyrights may follow
#
# $HEADER$

CC = gcc
CCOPT = -g -Wall -O3 -march=native -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DWITHPAPI -DDHAMT=$(DHAMT)  -DDDA4MT=$(DDA4MT) 
LD = gcc
LDOPT = 
LIBS =  -llapacke -lopenblas -lpapi

all: minitest_haar minitest_daub4

minitest_haar: minitest_haar.o haar.o matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

minitest_daub4: minitest_daub4.o daub4.o matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CCOPT) -c $<

clean:
	@rm -f *o minitest_haar minitest_daub4

.PHONY: clean
