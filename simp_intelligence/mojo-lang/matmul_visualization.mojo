from layout.tensor_builder import LayoutTensorBuild as tensor_builder
from layout.layout_tensor import Layout, LayoutTensor

fn example_tensor[rows: Int, columns: Int, base: Int =100]() -> LayoutTensor[
    DType.int32,
    Layout.row_major(rows, columns),
    MutableAnyOrigin,
]:
    var t = tensor_builder[DType.int32]().row_major[rows, columns]().alloc()
    for r in range(rows):
        for c in range(columns):
            t[r, c] = (r + 1) * (1000 + base) + c
    return rebind[LayoutTensor[DType.int32, Layout.row_major(rows, columns), MutableAnyOrigin]](t)

fn main():
    var A = example_tensor[8, 8, 100]()
    var B = example_tensor[8, 16, 200]()
    var C = example_tensor[8, 16, 0]()
    print(A, end="\n\n")
    print(B, end="\n\n")
    print(C, end="\n\n")

    M, N = A.dim[0](), B.dim[1]() # (8, 16)
    alias BM = 4
    alias BN = 4
    alias BK = 2
    alias TM = 2
    alias NUM_THREADS = (BM * BN) // TM
    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)) # (4, 2)
    block_dim=NUM_THREADS # 8
    tiled_register_matmul[C.dtype, A.layout, B.layout, C.layout, BM, BK, BN, TM, NUM_THREADS](
        A, B, C,
        block_idx_x=2, block_idx_y=1, thread_idx_x=5
    )

from math import ceildiv
fn tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, NUM_THREADS: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
        # simulating one thread
        block_idx_x: Int, block_idx_y: Int, thread_idx_x: Int,
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        var subtile_row = thread_idx_x // BN
        var subtile_col = thread_idx_x % BN

        var A_smem = tensor_builder[dtype]().row_major[BM, BK]().alloc()
        var B_smem = tensor_builder[dtype]().row_major[BK, BN]().alloc()

        var dst_tile = C.tile[BM, BN](block_idx_y, block_idx_x)
        var dst_subtile = dst_tile.tile[TM, 1](subtile_row, subtile_col)
        var dst_reg = tensor_builder[dtype]().layout[TM]().alloc()
        dst_reg.copy_from(dst_subtile) # copy the initial zeros

        for block in range(ceildiv(K, BK)):
            alias A_tile_layout = Layout.row_major(BM, BK)
            alias B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx_y, block)
            var B_tile = B.tile[BK, BN](block, block_idx_x)

            A_smem.copy_from(A_tile)
            B_smem.copy_from(B_tile)

            for k in range(BK):
                var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                var B_subtile = B_smem.tile[1, BN](k, 0)
                var B_element = B_subtile[0, subtile_col]

                for t in range(TM):
                    dst_reg[t] += A_subtile[t, 0] * B_element

        dst_subtile.copy_from(dst_reg)
        print(dst_subtile)
