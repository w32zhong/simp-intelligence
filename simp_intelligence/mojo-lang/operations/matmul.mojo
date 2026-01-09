import compiler
from gpu import (
    block_dim,
    block_idx,
    thread_idx,
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier
)
from gpu.memory import async_copy_wait_all
from gpu.host import DeviceBuffer, DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
)
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram_async,
)
from layout.tensor_builder import LayoutTensorBuild as tensor_builder
from utils import StaticTuple
from math import ceildiv


#@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn naive_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
    ):
        M = A.dim[0]()
        K = B.dim[0]()
        N = B.dim[1]()

        var row = block_dim.x * block_idx.x + thread_idx.x
        var col = block_dim.y * block_idx.y + thread_idx.y

        var dst_reg: C.element_type = 0

        if row < M and col < N:
            for k in range(K):
                dst_reg = dst_reg + A[row, k] * B[k, col]
            C[row, col] = dst_reg


fn coalescing_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
    ):
        M = A.dim[0]()
        K = B.dim[0]()
        N = B.dim[1]()

        # With this change, adjacent threads access values in the same row,
        # which are contiguous in memory.
        var row = block_dim.y * block_idx.y + thread_idx.y # slow
        var col = block_dim.x * block_idx.x + thread_idx.x # fast

        var dst_reg: C.element_type = 0

        if row < M and col < N:
            for k in range(K):
                dst_reg = dst_reg + A[row, k] * B[k, col]
            C[row, col] = dst_reg


fn zero_out[dtype: DType, layout: Layout](
        ctx: DeviceContextPtr,
        tensor: LayoutTensor[dtype, layout, MutableAnyOrigin]
    ) raises:
        bytes = tensor.shape[0]() * tensor.shape[1]()
        device_ctx = ctx.get_device_context()
        device_ctx.enqueue_memset(
            DeviceBuffer[dtype](
                device_ctx,
                rebind[UnsafePointer[Scalar[dtype]]](tensor.ptr),
                bytes,
                owning=False,
            ),
            0, # fill zeros
        )


fn tiled_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, NUM_THREADS: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        # Destination Tile of The Matrix C
        #      BN
        #    ______
        #   |      |
        # BM|      |
        #   | *    |   *: thread_idx
        #    ------
        var tile_row = thread_idx.x // BN
        var tile_col = thread_idx.x % BN

        var A_smem = tensor_builder[dtype]().row_major[BM, BK]().shared().alloc()
        var B_smem = tensor_builder[dtype]().row_major[BK, BN]().shared().alloc()

        var dst_tile = C.tile[BM, BN](block_idx.y, block_idx.x)
        var dst_reg: C.element_type = 0

        for block in range(ceildiv(K, BK)):
            alias A_tile_layout = Layout.row_major(BM, BK)
            alias B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            # copy A, B tiles from global memory to shared memory
            copy_dram_to_sram_async[thread_layout=A_tile_layout](A_smem, A_tile)
            copy_dram_to_sram_async[thread_layout=B_tile_layout](B_smem, B_tile)

            async_copy_wait_all() # this only means "the copy this thread launched is done"
            barrier() # this is needed to avoid stale contents @ smem[tile_row or tile_col]

            for k in range(BK):
                dst_reg += A_smem[tile_row, k] * B_smem[k, tile_col]

            barrier() # thread waiting before loading the next tiles

        dst_tile[tile_row, tile_col] += dst_reg


def tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, COMPUTE_THREADS: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
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

        dst_reg.copy_from(dst_subtile)

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(BM, BK)
            comptime B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            A_tile.log(filename='A_tile', block=block)
            B_tile.log(filename='B_tile', block=block)

            copy_dram_to_sram_async[thread_layout=A_tile_layout](A_smem, A_tile)
            copy_dram_to_sram_async[thread_layout=B_tile_layout](B_smem, B_tile)

            async_copy_wait_all()

            if participates_in_compute:
                for k in range(BK):
                    var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                    var B_element = B_smem[k, subtile_col]
                    for t in range(TM):
                        product = A_subtile[t, 0] * B_element
                        dst_reg[t] += product

            barrier()

        if participates_in_compute:
            dst_subtile.copy_from(dst_reg)


@compiler.register("my_matmul")
struct MyMatMul[algorithm: StaticString]:
    @staticmethod
    fn execute(
        raw_output: OutputTensor[rank=2],
        raw_A: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        raw_B: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        #print("algo: " + algorithm)
        device_ctx = ctx.get_device_context()

        A = raw_A.to_layout_tensor()
        B = raw_B.to_layout_tensor()
        output = raw_output.to_layout_tensor()

        # although naive and coalescing kernels do not need this,
        # keep it here in order to make fair kernel comparisions.
        zero_out[output.dtype, output.layout](ctx, output)

        M = A.shape[0]()
        N = B.shape[1]()

        alias OPTIMIZED_BLOCK_SIZE = 16
        alias BM = OPTIMIZED_BLOCK_SIZE
        alias BN = OPTIMIZED_BLOCK_SIZE
        alias BK = OPTIMIZED_BLOCK_SIZE

        if algorithm == "naive":
            device_ctx.enqueue_function[
                naive_matmul[
                    output.dtype, A.layout, B.layout, output.layout
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(M, BM), ceildiv(N, BN)),
                block_dim=(BM, BN),
            )

        elif algorithm == "coalescing":
            device_ctx.enqueue_function[
                coalescing_matmul[
                    output.dtype, A.layout, B.layout, output.layout
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(BN, BM),
            )

        elif algorithm == "tiled":
            alias NUM_THREADS = BM * BN
            device_ctx.enqueue_function[
                tiled_matmul[
                    output.dtype, A.layout, B.layout, output.layout,
                    BM, BK, BN, NUM_THREADS
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=NUM_THREADS,
            )

        elif algorithm == "tiled_register":
            comptime TM = 4
            comptime COMPUTE_THREADS = (BM * BN) // TM
            comptime COPY_THREADS = max(BM * BK, BK * BN)
            comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)

            device_ctx.enqueue_function[
                tiled_register_matmul[
                    output.dtype, A.layout, B.layout, output.layout,
                    BM, BK, BN, TM, COMPUTE_THREADS
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=NUM_THREADS,
            )

        else:
            raise Error("Unknown algorithm:", algorithm)

        #device_ctx.synchronize()
