# This file is part of pawt.
#
# Makefile, to compile the example programs
#
# Copyright 2018 LIPN, CNRS UMR 7030, Université Paris 13, 
#                Sorbonne-Paris-Cité. All rights reserved.
# Author: see AUTHORS
# Licence: GPL-3.0, see COPYING for details.
#
# Additional copyrights may follow
#
# $HEADER$

PAWT = -L.. -lpawt
CC = gcc
CCOPT = -g -Wall -O3
LD = gcc
LDOPT = 
LIBS =  -llapacke -lopenblas $(PAWT)	

all: solve_proto

solve_proto: solve_proto.o ../libpawt.so ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CCOPT) -c $<

clean:
	@rm -f *o solve_proto

.PHONY: clean
