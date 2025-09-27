## Quick start
```sh
uv sync
```

```sh
export PYTHONUNBUFFERED=1
uv run jupyter-notebook --autoreload .
```

## Convert a notebook to source file
```sh
uv pip install nbconvert
jupyter nbconvert --to script foo.ipynb
```

## Run a modified stdlib
```sh
uv pip install --pre modular==25.7.0.dev2025092205 \
  --index-url https://dl.modular.com/public/nightly/python/simple/
# clone the 25.7.0.dev2025092205 upstream code to ./modular
(cd ./modular && ./bazelw build //mojo/stdlib/stdlib)
MODULAR_MOJO_MAX_IMPORT_PATH=`pwd`/modular/bazel-bin mojo ./test.mojo
```

## Run Triton program
```sh
uv run python simp_intelligence/triton-dsl/vec_add.py
uv run python -m triton.tools.compile --kernel-name matmul_kernel --kernel-name add_kernel --signature "*fp32,*fp32,*fp32,i32,64" --grid=1024,1024,1024 ./simp_intelligence/triton-dsl/vec_add.py
rm -f *.[ch]
```
