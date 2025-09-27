.PHONY: clean all

#CFLAGS = -G -g # for debugging
CFLAGS = -lineinfo # for profiling

all: hello_world.out

%.out: %.cu Makefile
	nvcc $(CFLAGS) -o $@ $<

dump-%: %
	sudo ncu --set full -o $@ $*
	nsys profile -o $@ $*

clean:
	rm -f *.out dump-*
