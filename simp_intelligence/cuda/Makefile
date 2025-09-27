.PHONY: clean all

all: hello_world.out

%.out: %.cu
	nvcc -o $@ $<

clean:
	rm -f *.out
