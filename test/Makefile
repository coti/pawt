CC = gcc
CCOPT = -g -Wall -O3 -march=native -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -I/packages/papi/5.6.0/include/ -I$(HOME)/tools/include
LD = gcc
LDOPT = 
LIBS = -L$(HOME)/tools/lib   -lopenblas
BENCHS = -DWITHPAPI -L/packages/papi/5.6.0/lib


all: haar_test daub4_test

haar_test: haar_test.o ../haar.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

haar_bench: haar_bench.o ../haar.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS) $(BENCHS) -lpapi

daub4_test: daub4_test.o ../daub4.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

daub6_test: daub6_test.o ../daub6.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

daub4_bench: daub4_bench.o ../daub4.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS) $(BENCHS) -lpapi

%.o: %.c
	$(CC) $(CCOPT) -c $< $(BENCHS)

clean:
	@rm -f *o haar_test daub4_test daub4_bench haar_bench daub6_test

.PHONY: clean
