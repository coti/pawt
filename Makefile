CC = gcc
CCOPT = -Wall -O3
LD = gcc
LDOPT =
LIBS =  -llapacke -lopenblas

all: minitest_haar

minitest_haar: minitest_haar.o haar.o
	$(LD) $(LDOPT) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CCOPT) -c $<

clean:
	@rm -f *o minitest_haar

.PHONY: clean
