from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace
from random import random_si64
from memory import UnsafePointer, alloc
from . import block_idx, thread_idx


fn _get_address_space_name[addr: AddressSpace]() -> String:
    @parameter
    if addr == AddressSpace.GENERIC: return "AddressSpace.GENERIC"
    @parameter
    if addr == AddressSpace.GLOBAL: return "AddressSpace.GLOBAL"
    @parameter
    if addr == AddressSpace.SHARED: return "AddressSpace.SHARED"
    @parameter
    if addr == AddressSpace.LOCAL: return "AddressSpace.LOCAL"
    return "AddressSpace(Unknown)"


struct LoggedTensor[
    mut: Bool,
    //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut=mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    layout_int_type: DType = DType.int32,
    linear_idx_type: DType = DType.int32,
    masked: Bool = False,
]:
    alias ImplType = LayoutTensor[
        dtype, layout, origin,
        address_space=AddressSpace.GENERIC, # fixed b/c we run on CPU
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    alias OriginCastType[
        mut: Bool,
        origin: Origin[mut=mut],
    ] = LoggedTensor[
        Self.dtype,
        Self.layout,
        origin,
        address_space = Self.address_space,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
        masked = Self.masked,
    ]
    alias _AsMut = Self.OriginCastType[True, _]

    var impl: Self.ImplType
    var name: String
    var origin_x: Int
    var origin_y: Int

    fn __init__(out self,
                impl: Self.ImplType,
                name: String = "Tensor",
                origin_x: Int = 0,
                origin_y: Int = 0):
        self.impl = impl
        self.name = name
        self.origin_x = origin_x
        self.origin_y = origin_y

    fn print(read self, max_rows: Int = 12, max_cols: Int = 8, grid: Int = 4) raises:
        print("name={} layout={} coordinates=({}, {})".format(
            self.name, String(self.impl.runtime_layout),
            self.origin_x, self.origin_y
        ), end=":\n")
        var n_rows = self.impl.shape[0]()
        var n_cols = self.impl.shape[1]()
        var last_row: Int = 0
        for row in range(min(n_rows, max_rows)):
            if row % grid == 0:
                print("*", end="")
                for _ in range(min(n_cols, max_cols)):
                    print("-----".rjust(5), end="")
                print()
            var last_col: Int = 0
            for col in range(min(n_cols, max_cols)):
                if col % grid == 0: print("|", end="")
                print(
                    String(self.impl[row, col]).rjust(5),
                    end=""
                )
                last_col = col
            if last_col + 1 == max_cols and last_col + 1 != n_cols:
                print(' ...')
            else:
                print()
            last_row = row
        if last_row + 1 == max_rows and last_row + 1 != n_rows:
            for _ in range(min(n_cols, max_cols)):
                print("...".rjust(5), end="")
        print()

    fn log(read self, filename: StaticString = "tmp", **kwargs: Int) raises:
        var x = self.origin_x
        var y = self.origin_y
        var n_rows = self.impl.shape[0]()
        var n_cols = self.impl.shape[1]()
        var json_contents = """
            "thread_id.x": {},
            "thread_id.y": {},
            "thread_id.z": {},
            "block_id.x": {},
            "block_id.y": {},
            "block_id.z": {},
            "n_rows": {},
            "n_cols": {},
            "x": {},
            "y": {},
        """.format(
            thread_idx.x, thread_idx.y, thread_idx.z,
            block_idx.x, block_idx.y, block_idx.z,
            n_rows, n_cols, x, y
        ).replace(
            " ", ""
        ).replace(
            "\n", ""
        )

        for item in kwargs.items():
            json_contents += '"{}":{},'.format(item.key, item.value)

        with open(filename + ".log", "a") as fh:
            fh.write('{' + json_contents.rstrip(',') + '}\n'
            )

    fn dim[idx: Int](self) -> Int:
        return self.impl.dim[idx]()

    @always_inline
    @staticmethod
    fn stack_allocation[
        *, stack_alignment: Int = Self.ImplType.alignment
    ]() -> LoggedTensor[
            dtype,
            layout,
            MutAnyOrigin,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
            masked=masked
        ]:
        return LoggedTensor(
            Self.ImplType.stack_allocation[stack_alignment=stack_alignment](),
            _get_address_space_name[address_space]()
        )

    @always_inline
    fn tile[
        *tile_sizes: Int
    ](self, x: Int, y: Int) raises -> LoggedTensor[
        dtype,
        Self.ImplType.TileType[*tile_sizes].layout,
        origin,
        address_space=Self.ImplType.TileType[*tile_sizes].address_space,
        layout_int_type=Self.ImplType.TileType[*tile_sizes].layout_int_type,
        linear_idx_type=Self.ImplType.TileType[*tile_sizes].linear_idx_type,
        masked=Self.ImplType.TileType[*tile_sizes].masked,
    ]:
        var tiled_view = self.impl.tile[*tile_sizes](x, y)
        var new_name = "{}.tile[{}, {}]({}, {})".format(
            self.name, tile_sizes[0], tile_sizes[1], x, y
        )

        var new_origin_x = self.origin_x + x * tile_sizes[0]
        var new_origin_y = self.origin_y + y * tile_sizes[1]

        alias NewT = Self.ImplType.TileType[*tile_sizes]
        return LoggedTensor[
            dtype,
            NewT.layout,
            origin,
            address_space=NewT.address_space,
            layout_int_type=NewT.layout_int_type,
            linear_idx_type=NewT.linear_idx_type,
            masked=NewT.masked,
        ](tiled_view, new_name, new_origin_x, new_origin_y)

    fn copy_from(mut self: Self._AsMut, other: LoggedTensor):
        self.origin_x = other.origin_x
        self.origin_y = other.origin_y
        return self.impl.copy_from(other.impl)

    fn __getitem__(self, x: Int, y: Int) -> Self.ImplType.element_type:
        return self.impl[x, y]

    fn __getitem__(self, x: Int) -> Self.ImplType.element_type:
        return self.impl[x]

    fn __setitem__(self, x: Int, val: Self.ImplType.element_type) where Self.mut:
        self.impl[x] = val

    fn __setitem__(self, x: Int, y: Int, val: Self.ImplType.element_type) where Self.mut:
        self.impl[x, y] = val


fn example_logged_tensor[
        rows: Int, cols: Int
    ](name: String) -> LoggedTensor[
        DType.float32,
        Layout.row_major(rows, cols),
        MutAnyOrigin,
        address_space=AddressSpace.GENERIC,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=False
    ]:
    comptime buf_size = rows * cols
    var ptr = alloc[Float32](buf_size)
    for i in range(buf_size):
        ptr[i] = Float32(random_si64(-5, 5))
    comptime layout = Layout.row_major(rows, cols)
    var tensor = LayoutTensor[
        DType.float32,
        layout,
        MutAnyOrigin,
        address_space=AddressSpace.GENERIC,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=False
    ](UnsafePointer[Float32, MutAnyOrigin](ptr))
    return LoggedTensor(tensor, name)
