CC = gcc
CCOPT = -g -Wall -O3 -march=native -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store
LD = gcc
LDOPT = 
LIBS =  -llapacke -lopenblas
BENCHS = -DWITHPAPI


all: haar_test daub4_test

haar_test: haar_test.o ../haar.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

daub4_test: daub4_test.o ../daub4.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

daub4_bench: daub4_bench.o ../daub4.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS) $(BENCHS) -lpapi

%.o: %.c
	$(CC) $(CCOPT) -c $< $(BENCHS)

clean:
	@rm -f *o haar_test daub4_test daub4_bench

.PHONY: clean
