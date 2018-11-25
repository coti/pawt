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
