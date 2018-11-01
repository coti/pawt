## Possible DHAMT values:
# dhamt2_initial: initial, naive
# dhamt2_loop: loop order optimization
# dhamt2_avx: avx, using set_pd to load the data
# dhamt2_avx_gather: avx, using gather_pd
# dhamt2_sse: sse implementation for older process (pre-Haswell)
# dhamt2_fma: fused multiply-and-add implementation
DHAMT=dhamt2_avx

CC = gcc
CCOPT = -g -Wall -O3 -march=native -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DWITHPAPI -DDHAMT=$(DHAMT)
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
