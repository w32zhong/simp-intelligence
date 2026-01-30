import matplotlib.pyplot as plt


def default_color_map(index):
    colors = plt.cm.tab20b.colors[:]
    return colors[index % len(colors)]


def product(_tuple):
    if not isinstance(_tuple, tuple):
        return _tuple
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

    def __repr__(self):
        return f'{self.shape}:{self.stride}'

    def crd2idx(self, coordinates):
        offset = 0
        for c, s in zip(coordinates, self.stride):
            offset += c * s
        return offset

    def coordinates(self, shape=None, prefix_crd=tuple()):
        if shape is None:
            shape = self.shape
        return product(prefix_crd + tuple(tuple(range(s)) for s in shape))

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

    l1 = Layout(shape=(2, 3, 4))
    print(l1)
    print(list(l1.coordinates()))
    print(l1.permute(2, 0, 1))
    l1.visualize()
    plt.show()
