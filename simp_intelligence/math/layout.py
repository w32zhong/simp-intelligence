# Reference: https://github.com/HanGuo97/hilt/blob/main/hilt/pycute_utils.py
import math
import itertools
import matplotlib.pyplot as plt


def default_color_map(index):
    colors = plt.cm.tab20b.colors[:]
    return colors[index % len(colors)]


class Layout:
    def __init__(self, shape, stride):
        self.shape = shape
        self.stride = stride

    def __repr__(self):
        return f'{self.shape}:{self.stride}'

    def __getitem__(self, index):
        offset = 0
        for i, s in zip(index, self.stride):
            offset += i * s
        return offset

    def indices(self):
        # Product Reference
        #def product(*args):
        #    pools = [tuple(pool) for pool in args]
        #    result = [[]]
        #    for pool in pools:
        #        result = [x + [y] for x in result for y in pool]
        #    for prod in result:
        #        yield tuple(prod)
        for idx in itertools.product(*[range(d) for d in self.shape]):
            yield idx

    def visualize(self, color_map=default_color_map):
        if len(self.shape) == 3:
            fig, axes = plt.subplots(self.shape[0])
            for i in range(self.shape[0]):
                indices = itertools.product((i, ), *[range(d) for d in self.shape[1:]])
                self.visualize_2D(axes[i], indices, color_map)
        elif len(self.shape) == 2:
            fig, ax = plt.subplots()
            self.visualize_2D(ax, *self.shape, color_map)
        else:
            raise NotImplementedError
        plt.tight_layout()

    def visualize_2D(self, ax, indices, color_map):
        M, N = self.shape[-2], self.shape[-1]
        for idx in indices:
            offset = self[idx]
            color = color_map(offset)
            label = f'{offset}'

            m, n = idx[-2], idx[-1]
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

        # Add row labels (m indices) - positioned to the left
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

        # Add column labels (n indices) - positioned at the top
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


if __name__ == "__main__":
    layout = Layout(shape=(3, 2, 8), stride=(16, 8, 1))
    layout.visualize()
    plt.show()
    layout.permute(1, 2, 0)
    layout.visualize()
    plt.show()
