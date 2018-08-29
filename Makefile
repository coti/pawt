CC = gcc
CCOPT = -g -Wall -O3 -march=native -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DWITHPAPI 
LD = gcc
LDOPT =
LIBS =  -llapacke -lopenblas -lpapi

all: minitest_haar

minitest_haar: minitest_haar.o haar.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CCOPT) -c $<

clean:
	@rm -f *o minitest_haar

.PHONY: clean
