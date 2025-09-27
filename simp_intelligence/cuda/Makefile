.PHONY: clean all

#CFLAGS = -G -g # for debugging
CFLAGS = -lineinfo # for profiling

all: hello_world.out

%.out: %.cu Makefile
	nvcc $(CFLAGS) -o $@ $<

clean:
	rm -f *.out
