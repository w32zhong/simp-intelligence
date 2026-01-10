from layout import Layout
from visualization_utils import LoggedTensor as LayoutTensor
from visualization_utils import block_idx, thread_idx, barrier, warp_id, WARP_SIZE
from visualization_utils import example_logged_tensor, clear_log_files

from math import ceildiv
from layout.tensor_core import TensorCore
from utils.index import Index


def tensor_core_matmul_kernel[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, WM: Int, WN: Int, MMA_M: Int, MMA_K: Int, MMA_N: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
    ):
        var M = A.shape[0]()
        var K = B.shape[0]()
        var N = B.shape[1]()

        warp_y = warp_id() // UInt(BN // WN)
        warp_x = warp_id() % UInt(BN // WN)

        var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x)
                         .tile[WM, WN](Int(warp_y), Int(warp_x))
        C.tile[BM, BN](block_idx.y, block_idx.x).log(filename='C_tile')
        C_warp_tile.log(filename='C_warp_tile')

        var A_smem_tile = LayoutTensor[
            dtype,
            Layout.row_major(BM, BK),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var B_smem_tile = LayoutTensor[
            dtype,
            Layout.row_major(BK, BN),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var C_reg_tile = LayoutTensor[
            dtype,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation().fill(0)

        for block in range(ceildiv(K, BK)):
            A_dram_tile = A.tile[BM, BK](Int(block_idx.y), block)
            B_dram_tile = B.tile[BK, BN](block, Int(block_idx.x))

            A_smem_tile.copy_from(A_dram_tile)
            B_smem_tile.copy_from(B_dram_tile)
            A_smem_tile.log(filename='A_tile', block=block)
            B_smem_tile.log(filename='B_tile', block=block)

            A_warp_tile = A_smem_tile.tile[WM, BK](Int(warp_y), 0)
            B_warp_tile = B_smem_tile.tile[BK, WN](0, Int(warp_x))
            A_warp_tile.log(filename='A_warp_tile', block=block)
            B_warp_tile.log(filename='B_warp_tile', block=block)

            for mma_k in range(BK // MMA_K):
                for mma_m in range(WM // MMA_M):
                    for mma_n in range(WN // MMA_N):
                        A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                        B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)
                        A_mma_tile.log(filename='A_mma_tile', mma_m=mma_m, mma_k=mma_k)
                        B_mma_tile.log(filename='B_mma_tile', mma_k=mma_k, mma_n=mma_n)

        for mma_m in range(WM // MMA_M):
            for mma_n in range(WN // MMA_N):
                var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
                C_mma_tile.log(filename='C_mma_tile', mma_m=mma_m, mma_n=mma_n)


fn main() raises:
    clear_log_files()

    alias M = 64
    alias K = 48
    alias N = 64
    A = example_logged_tensor[M, K]("A")
    B = example_logged_tensor[K, N]("B")
    C = example_logged_tensor[M, N]("C")

    comptime OPTIMIZED_BLOCK_SIZE = 16
    comptime BM = OPTIMIZED_BLOCK_SIZE
    comptime BN = OPTIMIZED_BLOCK_SIZE
    comptime BK = OPTIMIZED_BLOCK_SIZE
    comptime WM = 8
    comptime WN = WARP_SIZE
    comptime MMA_M = 4
    comptime MMA_N = 4
    comptime MMA_K = 2
    comptime NUM_WARPS = (BM // WM) * (BN // WN)

    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    grid_dim = (ceildiv(N, BN), ceildiv(M, BM))
    block_dim = NUM_WARPS * WARP_SIZE

    for block_id_x in range(grid_dim[0]):
        for block_id_y in range(grid_dim[1]):
            for thread_id in range(block_dim):
                block_idx.set_dim3(block_id_x, block_id_y)
                thread_idx.set_dim3(thread_id)

                print(block_id_x, block_id_y, thread_id)
                tensor_core_matmul_kernel[
                    DType.float32,
                    Layout.row_major(M, K),
                    Layout.row_major(K, N),
                    Layout.row_major(M, N),
                    BM, BK, BN,
                    WM, WN,
                    MMA_M, MMA_K, MMA_N
                ](A, B, C)
