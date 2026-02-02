import ast
import math
import itertools
import matplotlib.pyplot as plt


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


def flat(t):
    if isinstance(t, tuple):
        if len(t) == 0:
            return ()
        else:
            return tuple(ttt for tt in t for ttt in flat(tt))
    else:
        return (t,)


def coalesce(layout):
    result_shape  = [1]
    result_stride = [0]
    for (shape, stride) in zip(flat(layout.shape), flat(layout.stride)):
        if shape == 1:
            # skip current trival shape
            # s0:d0  ++  1:d1  =>  s0:d0
            continue

        elif result_shape[-1] == 1:
            # skip previous trival shape
            # 1:d0  ++  s1:d1  =>  s1:d1
            result_shape[-1]  = shape
            result_stride[-1] = stride

        elif result_shape[-1] * result_stride[-1] == stride:
            # coalesce continous dimensions
            # s0:d0  ++  s1:s0*d0  =>  s0*s1:d0
            result_shape[-1] = result_shape[-1] * shape

        else:
            # nothing can be simplified
            # s0:d0  ++  s1:d1  =>  (s0,s1):(d0,d1)
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        return Layout(tuple(result_shape), tuple(result_stride))


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

    @classmethod
    def from_concate(cls, *layouts):
        shape = itertools.chain(layout.shape for layout in layouts)
        stride = itertools.chain(layout.stride for layout in layouts)
        return Layout(tuple(shape), tuple(stride))

    def __repr__(self):
        return f'{self.shape}:{self.stride}'

    def __len__(self):
        if isinstance(self.shape, tuple):
            return len(self.shape)
        else:
            return 1

    def rank(self):
        return len(self)

    def __getitem__(self, i):
        if isinstance(self.shape, tuple):
            return Layout(self.shape[i], self.stride[i])
        else:
            return Layout(self.shape, self.stride)

    def size(self): # Size of the domain
        return product(self.shape)

    @staticmethod
    def max_coordinates(shape):
        if isinstance(shape, int):
            return shape - 1
        elif isinstance(shape, tuple):
            return tuple(Layout.max_coordinates(s) for s in shape)
        else:
            raise TypeError(f"Unsupported type: {type(shape)}")

    @staticmethod
    def coordinates(shape):
        if isinstance(shape, int):
            yield from range(shape)
        elif isinstance(shape, tuple):
            for c in cartesian_product(tuple(
                tuple(Layout.coordinates(s)) for s in shape
            )):
                yield c
        else:
            raise TypeError(f"Unsupported type: {type(shape)}")

    def crd2idx(self, crd):
        if isinstance(crd, tuple):
            offset = 0
            for c, s, d in zip(crd, self.shape, self.stride):
                offset += Layout(s, d).crd2idx(c)
            return offset
        else:
            return crd * self.stride

    def cosize(self): # Size of the codomain
        max_crd = Layout.max_coordinates(self.shape)
        return self.crd2idx(max_crd) + 1

    def idx2crd(self, idx):
        if isinstance(self.shape, tuple):
            return tuple(Layout(s, d).idx2crd(idx) for s, d in zip(self.shape, self.stride))
        else:
            return (idx // self.stride) % self.shape

    def visualize(self, title=None, *, size_pad=2.0, size_scaler=0.5, color_map=default_color_map):
        if len(self) == 3:
            figsize = (
                size_pad + self[-1].size() * size_scaler,
                size_pad + (self[-2].size() * self[-1].size()) * size_scaler
            )
            fig, axes = plt.subplots(self[0].size(), figsize=figsize)
            for idx in range(self[0].size()):
                if self[0].size() == 1:
                    self.visualize_2D_or_1D(axes, idx, color_map)
                else:
                    self.visualize_2D_or_1D(axes[idx], idx, color_map)

        elif len(self) == 2:
            figsize = (
                size_pad + self[1].size() * size_scaler,
                size_pad + self[0].size() * size_scaler
            )
            fig, ax = plt.subplots(figsize=figsize)
            self.visualize_2D_or_1D(ax, None, color_map)

        elif len(self) == 1:
            figsize = (
                size_pad + self[0].size() * size_scaler,
                size_pad + 1 * size_scaler
            )
            fig, ax = plt.subplots(figsize=figsize)
            self.visualize_2D_or_1D(ax, None, color_map)

        else:
            raise NotImplementedError

        fig.suptitle((f'{title} = ' if title else '') + str(self))
        plt.tight_layout()
        return self

    def visualize_2D_or_1D(self, ax, z, color_map, debug=False):
        is_1D = (len(self) == 1)
        N = self[-1].size()
        if is_1D:
            M = 1
            _2D = Layout(shape=(1, self[-1].shape))
            _2D_flat = Layout((1, self[-1].size()))
        else:
            M = self[-2].size()
            _2D = Layout(shape=self[-2:].shape)
            _2D_flat = Layout((self[-2].size(), self[-1].size()))

        m_axis_ticks, n_axis_ticks = dict(), dict()
        for m in range(M):
            for n in range(N):
                idx = _2D_flat.crd2idx((m, n))
                _2D_crd = _2D.idx2crd(idx)

                if z is None:
                    if is_1D:
                        crd = _2D_crd[-1]
                        if isinstance(crd, tuple) and len(crd) != len(self):
                            crd = (crd,)
                    else:
                        crd = _2D_crd
                else:
                    crd = (z, *_2D_crd)

                offset = self.crd2idx(crd)
                if debug: print(crd, '->', (m, n), '->', offset)

                color = color_map(offset)
                label = f'{offset}'
                m_axis_ticks[m] = 0 if is_1D else crd[-2]
                n_axis_ticks[n] = crd[-1] if isinstance(crd, tuple) else crd
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
                    fontsize=8, fontweight="bold", color="black"
                )

        # Add row labels
        for m in range(M):
            ax.text(
                -0.3,
                M - m - 0.5,
                str(m_axis_ticks[m]),
                ha="right",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        # Add column labels
        for n in range(N):
            ax.text(
                n + 0.5,
                M + 0.3,
                str(n_axis_ticks[n]),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
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

    def composite(self, other, by_mode=False):
        if isinstance(other.shape, tuple):
            if by_mode:
                cat = (self[i].composite(other_i) for i, other_i in enumerate(other))
            else:
                cat = (self.composite(other_i) for other_i in other)
            return Layout.from_concate(*cat)
        else:
            result_shape = []
            result_stride = []
            remain_shape = other.shape
            remain_stride = other.stride
            for cur_shape, cur_stride in zip(flat(self.shape)[:-1], flat(self.stride)[:-1]):
                # Example   A_shape:A_stride / B_stride
                #         = (3,6,2,8):(w,x,y,z) / 72
                #         = (ceil[3/72], ceil[6/24], ceil[2/4], ceil[8/2]):(72w, 24x, 4x, 2z)
                #         = (1, 1, 1, 4):(72w, 24x, 4x, 2z)
                new_shape = max(1, cur_shape // remain_stride) # "dividing out"
                new_shape = min(new_shape, remain_shape) # "modding out"
                new_stride = cur_stride * remain_stride # adjust stride

                remain_shape = remain_shape // new_shape
                remain_stride = math.ceil(remain_stride / cur_shape)

                result_shape.append(new_shape)
                result_stride.append(new_stride)

            result_shape.append(remain_shape)
            result_stride.append(remain_stride * flat(self.stride)[-1])

            if len(result_shape) == 1:
                return Layout(result_shape[0], result_stride[0])
            else:
                return Layout(tuple(result_shape), tuple(result_stride))

    def complement(self, max_idx=1, *, coalesce_result=True):
        result_shape = []
        result_stride = []
        last_idx = 1
        reindex = sorted(zip(flat(self.stride), flat(self.shape)))
        for stride, shape in reindex:
            # complement(Layout) = (d1, d2/(s1*d1), d3/(s2*d2), ...):(1, s1*d1, s2*d2, ...)
            result_shape.append(stride // last_idx)
            result_stride.append(last_idx)
            last_idx = shape * stride

        result_shape.append((max_idx + last_idx - 1) // last_idx) # ceil divide
        result_stride.append(last_idx)
        result = Layout(tuple(result_shape), tuple(result_stride))
        return coalesce(result) if coalesce_result else result

    def logical_divide(self, other, by_mode=False):
        if by_mode:
            cat = (self[i].logical_divide(other_i) for i, other_i in enumerate(other))
            return Layout.from_concate(*cat)
        else:
            # A ⊘ B := A o (B, ~B)
            compl = other.complement(self.size())
            cat = Layout.from_concate(other, compl)
            return self.composite(cat)

    def logical_product(self, other, by_mode=False):
        if by_mode:
            cat = (self[i].logical_product(other_i) for i, other_i in enumerate(other))
            return Layout.from_concate(*cat)
        else:
            # A ⊗ B := (A, ~A ∘ B)
            size = self.size() * other.cosize()
            compl = self.complement(size)
            compo = compl.composite(other)
            return Layout.from_concate(self, compo)

    def blocked_product(self, other):
        prod = self.logical_product(other)
        cat = []
        for zipped in zip(prod[0], prod[1]):
            inner_layout = Layout.from_concate(*zipped)
            cat.append(inner_layout)
        return Layout.from_concate(*cat)


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

    print(coalesce(Layout.from_string('2, (1,6): 1, (6, 2)')))
    print(coalesce(Layout.from_string('(3, 1): (2, 3)')))
    print(coalesce(Layout.from_string('(2, 3): (3, 1)'))) # not handled
    print(coalesce(Layout.from_string('(2, 3): (3, 1)').permute(1, 0))) # handled

    l1 = Layout(shape=(2, 3, 4))
    print(l1.permute(2, 0, 1))
    #l1.visualize()

    l2 = Layout.from_string('2,8')
    print(l2, l2.idx2crd(7))

    l3 = Layout.from_string('(8, 2):(2, 1)')
    for crd in Layout.coordinates(l3.shape):
        print(crd, '->', l3.crd2idx(crd), end=', ')
    print()

    l4 = Layout.from_string('((4, 2),):((1, 4),)')
    print(l4)
    print(tuple(Layout.coordinates(l4.shape)))
    #l4.visualize()

    l5 = Layout.from_string('(2,2),(2,2):(1,4),(2,8)')
    print(l5, l5.shape)
    #l5.visualize()

    # sparse layout => size !=  cosize
    l6 = Layout.from_string('((3, 3), 4):((1, 3), 10)')
    print(tuple(Layout.coordinates(l6.shape)))
    print(Layout.max_coordinates(l6.shape))
    print(l6.size(), l6.cosize())

    l7 = Layout.from_string('((2, (2, 2)), (2, (2, 2))):((1, (4, 16)), (2, (8, 32)))')
    #l7.visualize()

    l8 = Layout.from_string('((3, 2), (2, 5)):((1, 6), (3, 12))')
    #l8.visualize()

    A = Layout.from_string('(6,2):(8,2)')
    B = Layout.from_string('(4,3):(3,1)')
    composed = A.composite(B)
    print(composed)
    #A.visualize(); B.visualize(); composed.visualize()

    A = Layout.from_string('20:2')
    B = Layout.from_string('(5,4):(4,1)')
    composed = A.composite(B)
    print(composed)
    #A.visualize(); B.visualize(); composed.visualize()

    A = Layout.from_string('(10,2):(16,4)')
    B = Layout.from_string('(5,4):(1,5)')
    composed = A.composite(B)
    print(composed)
    #A.visualize(); B.visualize(); composed.visualize()

    A = Layout.from_string('(12, (4, 8)):(59, (13, 1))')
    B = Layout.from_string('(3,8):(4,2)')
    composed = A.composite(B, by_mode=True)
    print(composed)
    #A.visualize(); B.visualize(); composed.visualize()

    A = Layout.from_string('(12, (4, 8)):(59, (13, 1))')
    B = Layout.from_string('(3,8):(1,1)')
    composed = A.composite(B, by_mode=True)
    print(composed)
    #A.visualize(); B.visualize(); composed.visualize()

    def test_complement(string, cosize=24, coalesce=True, visualize=False):
        base = Layout.from_string(string)
        comp = base.complement(cosize, coalesce_result=coalesce)
        print('~', base, '=', comp)
        full = Layout.from_concate(base, comp)
        assert full.cosize() == cosize
        if visualize:
            base.visualize('Base')
            comp.visualize('Complement')
            full.visualize('Full')

    test_complement('4:1',         coalesce=True, visualize=False)
    test_complement('6:4',         coalesce=True, visualize=False)
    test_complement('(4,6):(1,4)', coalesce=True, visualize=False)
    test_complement('4:2',         coalesce=True, visualize=False)
    test_complement('(2,4):(1,6)', coalesce=True, visualize=False)
    test_complement('(2,2):(1,6)', coalesce=True, visualize=False)

    #A = Layout.from_string('((4,2,3),):((2,1,8),)').visualize()
    #A.logical_divide(Layout.from_string('4:2')).visualize()

    A = Layout.from_string('(9,(4,8)):(59,(13,1))').visualize()
    C = A.logical_divide(Layout.from_string('3,(2,4):3,(1,8)'), by_mode=True)
    C.visualize()

    #A = Layout.from_string('((2, 2),):((4, 1),)') #.visualize()
    #C = A.logical_product(Layout.from_string('6:1'))
    #print(C); #C.visualize()

    #A = Layout.from_string('((2, 2),):((4, 1),)') #.visualize()
    #C = A.logical_product(Layout.from_string('(4,2):(2,1)'))
    #print(C); #C.visualize()

    #A = Layout.from_string('(2, 5):(5, 1)') #.visualize()
    #C = A.logical_product(Layout.from_string('(3, 4):(5, 6)'), by_mode=True)
    #print(C); #C.visualize()

    #A = Layout.from_string('(2, 2):(2, 1)') #.visualize()
    #C = A.blocked_product(Layout.from_string('(2, 3):(3, 1)'))
    #print(C); C.visualize()

    plt.show()
