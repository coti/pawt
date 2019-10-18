## Possible DHAMT values:
# dhamt2_initial: initial, naive
# dhamt2_loop: loop order optimization
# dhamt2_avx: avx, using set_pd to load the data
# dhamt2_avx_gather: avx, using gather_pd
# dhamt2_sse: sse implementation for older process (pre-Haswell)
# dhamt2_fma: fused multiply-and-add implementation
# dhamt2_fma_reuse: fused multiply-and-add implementation using indermediate AVX registers rather than W matrix
# dhamt2_fma512_reuse: fused multiply-and-add implementation using indermediate AVX registers rather than W matrix with AVX512
## Possible DHIMT values:
# dhimt2_initial
# dhimt2_fma
# dhimt2_fma_gather
# dhimt2_fma_reuse
# dhimt2_fma512_reuse

DHAMT=dhamt2_fma_reuse
DHIMT=dhimt2_fma_reuse

# dda4mt2_initial
# dda4mt2_loop
# dda4mt2_avx
# dda4mt2_avx_gather
# dda4mt2_fma
# dda4mt2_fma2
# dda4mt2_fma_reuse
# dda4mt2_fma2_reuse
# dda4mt2_fma2_reuse_gather
# dda4mt2_fma512_reuse

## Possible DDI4MT values:
# dhimt2_loop
# dhimt2_avx
# dhimt2_avx_gather
# dhimt2_fma
# dhimt2_fma_gather
# dhimt2_fma2
# dhimt2_fma2_gather
# ddi4mt2_fma_reuse
# ddi4mt2_fma512_reuse

DDA4MT=dda4mt2_fma512_reuse
DDI4MT=ddi4mt2_fma512_reuse



DDA6MT=dda6mt2_initial

DDA8MT=dda8mt2_initial

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

ARCHOPT = -march=native
ALIGNOPT = -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store 
CC = gcc
CCOPT = -g -Wall -O3 -DWITHPAPI $(ARCHOPT) $(ALIGNOPT) -DDHAMT=$(DHAMT) -DDHIMT=$(DHIMT) -DDDA4MT=$(DDA4MT) -DDDI4MT=$(DDI4MT) -DDDA6MT=$(DDA6MT) -DDDA8MT=$(DDA8MT) -I$(HOME)/tools/include  -I/packages/papi/5.6.0/include
LD = gcc
LDOPT = 
LIBS = -L$(HOME)/tools/lib -lopenblas -L/packages/papi/5.6.0/lib -lpapi

all: minitest_daub6

minitest_haar: minitest_haar.o haar.o matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

minitest_daub4: minitest_daub4.o daub4.o matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

minitest_daub6: minitest_daub6.o daub6.o matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

minitest_daub8: minitest_daub8.o daub8.o matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

libpawt.so: haar.o daub4.c daub6.o
	$(LD) $(LDOPT) $(ARCHOPT) -shared -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CCOPT) -c $<

clean:
	@rm -f *.o libpawt.so minitest_haar minitest_daub4 minitest_daub6 minitest_daub8

.PHONY: clean
