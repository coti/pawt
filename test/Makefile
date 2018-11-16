CC = gcc
CCOPT = -g -Wall -O3 -march=native -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store
LD = gcc
LDOPT = 
LIBS =  -llapacke -lopenblas 


all: haar_test daub4_test

haar_test: haar_test.o ../haar.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

daub4_test: daub4_test.o ../daub4.o ../matrices.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CCOPT) -c $<

clean:
	@rm -f *o haar_test daub4_test

.PHONY: clean
