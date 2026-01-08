from layout import Layout
from visualization_utils import LoggedTensor as LayoutTensor
from visualization_utils import block_idx, thread_idx, barrier
from visualization_utils import example_logged_tensor, clear_log_files

from math import ceildiv


def tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, COMPUTE_THREADS: Int,
        NUM_THREADS: Int, version: StaticString,
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        var subtile_row = thread_idx.x // BN
        var subtile_col = thread_idx.x % BN
        var max_subtile_rows = BM // TM
        var participates_in_compute = (
            subtile_row < max_subtile_rows and
            thread_idx.x < COMPUTE_THREADS
        )

        var A_smem = LayoutTensor[
            dtype,
            Layout.row_major(BM, BK),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var B_smem = LayoutTensor[
            dtype,
            Layout.row_major(BK, BN),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var dst_reg = LayoutTensor[
            dtype,
            Layout.row_major(TM),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        var dst_subtile = C.tile[BM, BN](block_idx.y, block_idx.x)
                           .tile[TM, 1](subtile_row, subtile_col)

        C.tile[BM, BN](block_idx.y, block_idx.x).log(filename='block_tile')
        dst_subtile.log(filename='thread_tile')

        dst_reg.copy_from(dst_subtile)

        barrier()

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(BM, BK)
            comptime B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            A_tile.log(filename='A_tile', block=block)
            B_tile.log(filename='B_tile', block=block)

            A_smem.copy_from(A_tile)
            B_smem.copy_from(B_tile)

            barrier()

            if participates_in_compute:
                for k in range(BK):
                    var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                    var B_subtile = B_smem.tile[1, BN](k, 0)

                    A_subtile.log(filename='A_subtile', block=block, k=k)
                    B_subtile.log(filename='B_subtile', block=block, k=k)

                    var B_element = B_subtile[0, subtile_col]

                    for t in range(TM):
                        product = A_subtile[t, 0] * B_element
                        #dst_reg[t] += product

            barrier()

        if participates_in_compute:
            dst_subtile.copy_from(dst_reg)


fn main() raises:
    clear_log_files()

    alias M = 48
    alias K = 56
    alias N = 24
    A = example_logged_tensor[M, K]("A")
    A.print()
    B = example_logged_tensor[K, N]("B")
    B.print()
    C = example_logged_tensor[M, N]("C")
    C.print()

    comptime OPTIMIZED_BLOCK_SIZE = 4
    comptime BM = OPTIMIZED_BLOCK_SIZE
    comptime BN = OPTIMIZED_BLOCK_SIZE
    comptime BK = OPTIMIZED_BLOCK_SIZE
    comptime TM = 2
    comptime COMPUTE_THREADS = (BM * BN) // TM
    comptime COPY_THREADS = max(BM * BK, BK * BN)
    comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)
    comptime version = 'whatever'

    grid_dim = (ceildiv(N, BN), ceildiv(M, BM))
    block_dim = NUM_THREADS
    for block_id_x in range(grid_dim[0]):
        for block_id_y in range(grid_dim[1]):
            for thread_id in range(NUM_THREADS):
                block_idx.set_dim3(block_id_x, block_id_y)
                thread_idx.set_dim3(thread_id)
                tiled_register_matmul[
                    DType.float32,
                    Layout.row_major(M, K),
                    Layout.row_major(K, N),
                    Layout.row_major(M, N),
                    BM, BK, BN, TM, COMPUTE_THREADS, NUM_THREADS, version,
                ](A, B, C)
