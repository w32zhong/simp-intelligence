import ast
import matplotlib.pyplot as plt
from functools import reduce


def default_color_map(index):
    colors = plt.cm.tab20b.colors[:]
    return colors[index % len(colors)]


def cartesian_product(_tuple):
    if not isinstance(_tuple, tuple):
        return [(_tuple,)]
    pools = []
    for _set in _tuple:
        if isinstance(_set, tuple):
            pools.append(_set)
        else:
            pools.append((_set,))
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    return [tuple(r) for r in result]


def product(_tuple):
    if not isinstance(_tuple, tuple):
        return _tuple
    prod = 1
    for elem in _tuple:
        prod *= product(elem)
    return prod


def prefix_product(_tuple, init=1):
    if isinstance(_tuple, tuple):
        if isinstance(init, tuple):
            assert len(_tuple) == len(init)
            return tuple(prefix_product(x, i) for x, i in zip(_tuple, init))
        else:
            r = []
            for v in _tuple:
                r.append(prefix_product(v, init))
                init = init * product(v)
            return tuple(r)
    else:
        return init


class Layout:
    def __init__(self, shape, stride=None):
        self.shape = shape
        self.stride = prefix_product(shape) if stride is None else stride

    @classmethod
    def from_string(cls, string):
        def _find_top_level_colon(s):
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ':' and depth == 0:
                    return i
            return -1

        def _recurr(s):
            s = s.strip()
            colon = _find_top_level_colon(s)
            if colon != -1:
                return cls(ast.literal_eval(s[:colon]), ast.literal_eval(s[colon+1:]))

            if s.startswith('(') and s.endswith(')'):
                return _recurr(s[1:-1])
            else:
                return cls(ast.literal_eval(s))

        return _recurr(string)

    def __repr__(self):
        return f'{self.shape}:{self.stride}'

    def __getitem__(self, i):
        if isinstance(self.shape, tuple):
            return Layout(self.shape[i], self.stride[i])
        else:
            assert i == 0
            return Layout(self.shape, self.stride)

    def size(self): # Size of the domain
        return product(self.shape)

    def crd2idx(self, crd):
        if isinstance(crd, tuple):
            assert len(crd) == len(self.shape)
            offset = 0
            for c, s, d in zip(crd, self.shape, self.stride):
                offset += Layout(s, d).crd2idx(c)
            return offset
        else:
            return crd * self.stride

    @staticmethod
    def max_coordinates(shape):
        if isinstance(shape, int):
            return shape - 1
        elif isinstance(shape, tuple):
            return tuple(Layout.max_coordinates(s) for s in shape)
        else:
            raise TypeError(f"Unsupported type: {type(shape)}")

    def cosize(self): # Size of the codomain
        max_crd = Layout.max_coordinates(self.shape)
        return self.crd2idx(max_crd) + 1

    def idx2crd(self, idx):
        if isinstance(self.shape, tuple):
            assert len(self.shape) == len(self.stride)
            return tuple(Layout(s, d).idx2crd(idx) for s, d in zip(self.shape, self.stride))
        else:
            return (idx // self.stride) % self.shape

    def coordinates(self, shape=None, prefix_crd=tuple()):
        if shape is None:
            shape = self.shape
        return cartesian_product(prefix_crd + tuple(tuple(range(s)) for s in shape))

    def visualize(self, color_map=default_color_map):
        if len(self.shape) == 3:
            fig, axes = plt.subplots(self.shape[0])
            for i in range(self.shape[0]):
                i_crds = self.coordinates(prefix_crd=(i,), shape=self.shape[1:])
                self.visualize_2D(axes[i], i_crds, color_map)
        elif len(self.shape) == 2:
            fig, ax = plt.subplots()
            self.visualize_2D(ax, self.coordinates(), color_map)
        else:
            raise NotImplementedError
        plt.tight_layout()

    def visualize_2D(self, ax, crds, color_map):
        M, N = self.shape[-2], self.shape[-1]
        for crd in crds:
            offset = self.crd2idx(crd)
            color = color_map(offset)
            label = f'{offset}'

            m, n = crd[-2], crd[-1]
            rect = plt.Rectangle(
                (n, M - m - 1), 1, 1,
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                n + 0.5, M - m - 0.5, label,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="black"
            )

        # Add row labels
        for m in range(M):
            ax.text(
                -0.3,
                M - m - 0.5,
                str(m),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

        # Add column labels
        for n in range(N):
            ax.text(
                n + 0.5,
                M + 0.3,
                str(n),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_xlim(0, N)
        ax.set_ylim(0, M)
        ax.set_aspect("equal")
        ax.axis("off")

    def permute(self, *dims):
        new_shape = tuple(self.shape[d] for d in dims)
        new_stride = tuple(self.stride[d] for d in dims)
        return Layout(new_shape, new_stride)


if __name__ == "__main__":
    #import cutlass.cute as cute
    #@cute.jit
    #def test():
    #    base = cute.make_layout(shape=(6, 8), stride=(8, 1))
    #    tiler = cute.make_layout(shape=(3, 2), stride=(1, 3))
    #    composed = cute.composition(base, tiler)
    #    print(composed) # (3,2):(8,24)

    #    base = cute.make_layout(shape=(6, 2), stride=(8, 2))
    #    tiler = cute.make_layout(shape=(4, 3), stride=(3, 1))
    #    composed = cute.composition(base, tiler)
    #    print(composed) # ((2,2),3):((24,2),8)
    #test()

    print(cartesian_product(
        (2, (3, 4), (5, (6, 8)))
    ))

    print(product(
        (2, (3, 4), (5, (6, 8)))
    ))

    print(prefix_product(
        (2, (3, 4, 2), (5, (6, 8), 6))
    ))

    l1 = Layout(shape=(2, 3, 4))
    print(l1)
    print(list(l1.coordinates()))
    print(l1.permute(2, 0, 1))
    l1.visualize()
    #plt.show()

    l2 = Layout.from_string('2,8')
    print(l2, l2.idx2crd(7))

    l3 = Layout.from_string('(8, 2):(2, 1)')
    for idx in range(l3.size()):
        print(idx, '->', l3.idx2crd(idx), end=', ')
    print()

    l4 = Layout.from_string('((4, 2),):((1, 4),)')
    print(l4[0])

    # sparse layout => size !=  cosize
    l5 = Layout.from_string('((3, 3), 4):((1, 3), 10)')
    print(Layout.max_coordinates(l5.shape))
    print(l5.size(), l5.cosize())
