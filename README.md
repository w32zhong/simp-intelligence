## Quick start
```sh
git submodule update --init --recursive
pixi install
pixi shell
pip install -e simp_intelligence/cute-dsl/hilt_src/
```

```sh
export PYTHONUNBUFFERED=1
jupyter-notebook --autoreload .
```

## Convert a notebook to source file
```sh
pip install nbconvert
jupyter nbconvert --to script foo.ipynb
```

## Run a modified stdlib
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
