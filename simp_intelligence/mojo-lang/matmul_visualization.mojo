from layout import Layout
from visualization_utils import LoggedTensor as LayoutTensor
from visualization_utils import block_idx, thread_idx, barrier
from visualization_utils import example_logged_tensor, clear_log_files

from math import ceildiv


def block_tiled_matrix_multiplication[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, TN: Int, COMPUTE_THREADS: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        var subtile_row = Int(thread_idx.x // Int(BN // TN))
        var subtile_col = Int(thread_idx.x % Int(BN // TN))

        var max_subtile_rows = BM // TM
        var max_subtile_cols = BN // TN
        var participates_in_compute = (
            subtile_row < max_subtile_rows and
            subtile_col < max_subtile_cols and
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
            Layout.row_major(TM, TN),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        var dst_subtile = C.tile[BM, BN](block_idx.y, block_idx.x)
                           .tile[TM, TN](subtile_row, subtile_col)

        dst_reg.copy_from(dst_subtile)

        C.tile[BM, BN](block_idx.y, block_idx.x).log(filename='block_tile')
        dst_subtile.log(filename='thread_tile')

        var A_reg = LayoutTensor[
            dtype,
            Layout(TM),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        var B_reg = LayoutTensor[
            dtype,
            Layout(TN),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(COMPUTE_THREADS // BK, BK)
            comptime B_tile_layout = Layout.row_major(BK, COMPUTE_THREADS // BK)

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
                    var B_subtile = B_smem.tile[1, TN](k, subtile_col)

                    A_reg.copy_from(A_subtile)
                    B_reg.copy_from(B_subtile)

                    A_reg.log(filename='A_subtile', block=block, k=k)
                    B_reg.log(filename='B_subtile', block=block, k=k)
                    #outer_product_acc(dst_reg, A_reg, B_reg)

            barrier()

        if participates_in_compute:
            dst_subtile.copy_from(dst_reg)


fn main() raises:
    clear_log_files()

    alias M = 28
    alias K = 48
    alias N = 12
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
    comptime TN = 2
    comptime COMPUTE_THREADS = (BM * BN) // (TM * TN)
    comptime COPY_THREADS = max(BM * BK, BK * BN)
    comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)

    grid_dim = (ceildiv(N, BN), ceildiv(M, BM))
    block_dim = NUM_THREADS
    for block_id_x in range(grid_dim[0]):
        for block_id_y in range(grid_dim[1]):
            for thread_id in range(block_dim):
                block_idx.set_dim3(block_id_x, block_id_y)
                thread_idx.set_dim3(thread_id)
                block_tiled_matrix_multiplication[
                    DType.float32,
                    Layout.row_major(M, K),
                    Layout.row_major(K, N),
                    Layout.row_major(M, N),
                    BM, BK, BN, TM, TN, COMPUTE_THREADS
                ](A, B, C)
