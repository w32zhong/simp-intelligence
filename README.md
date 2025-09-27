# About
This is my learning environment and playground for GPU-stack software and AI algorithms.

## Quick start
```sh
pixi install
pixi shell
git submodule update --init --recursive
pip install -e simp_intelligence/cute-dsl/hilt_src/
```

Verify system installations:
```sh
$ echo $CC
/simp-intelligence/.pixi/envs/default/bin/x86_64-conda-linux-gnu-cc
$ which nvcc
/simp-intelligence/.pixi/envs/default/bin/nvcc
$ nvcc --version
...
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

```sh
export PYTHONUNBUFFERED=1
jupyter-notebook --autoreload .
```

## Convert a notebook to a source file
```sh
pip install nbconvert
jupyter nbconvert --to script foo.ipynb
```

## Run a modified stdlib in Mojo
```sh
# clone the 25.7.0.dev2025092205 upstream code to ./modular
(cd ./modular && ./bazelw build //mojo/stdlib/stdlib)
MODULAR_MOJO_MAX_IMPORT_PATH=`pwd`/modular/bazel-bin mojo ./test.mojo
```

## Run Triton program
```sh
python simp_intelligence/triton-dsl/vec_add.py
# generate trans-compiled C files
python -m triton.tools.compile --kernel-name matmul_kernel --kernel-name add_kernel --signature "*fp32,*fp32,*fp32,i32,64" --grid=1024,1024,1024 ./simp_intelligence/triton-dsl/vec_add.py
rm -f *.[ch]
```

## Using CUDA
```sh
cd simp_intelligence/cuda
make clean && make
make dump-hello_world.out
ncu-ui dump-hello_world.out.ncu-rep
nsys-ui dump-hello_world.out.nsys-rep
# https://www.youtube.com/watch?v=F_BazucyCMw
```
