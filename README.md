## Quick Start
```sh
pixi install
pixi run jupyter-notebook --autoreload .
```

## Convert A Notebook to Source File
```sh
uv pip install nbconvert
jupyter nbconvert --to script foo.ipynb
```
