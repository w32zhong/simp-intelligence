import compiler
from gpu import (
    block_dim,
    block_idx,
    thread_idx,
    warp_id,
    WARP_SIZE,
    barrier
)
from gpu.memory import AddressSpace, async_copy_wait_all
from gpu.host import DeviceBuffer, DeviceContext, device_attribute
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram_async,
)
from layout.tensor_core import TensorCore
from utils.index import Index
from layout.math import outer_product_acc
from sys.info import simd_width_of, has_amd_gpu_accelerator
from math import ceildiv


fn naive_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
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
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
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


fn tiled_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, NUM_THREADS: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
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


fn tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, COMPUTE_THREADS: Int
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

        dst_reg.copy_from(dst_subtile)

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(BM, BK)
            comptime B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            # this copy_dram_to_sram_async requires a full grid of thread workers, i.e.,
            # COPY_THREADS = max(BM * BK, BK * BN).
            # Because inside copy_dram_to_sram_async, the logic looks roughly like this:
            # fn copy_dram_to_sram_async():
            #     my_id = thread_idx.x
            #     if my_id < total_elements in destination:
            #         copy
            copy_dram_to_sram_async[thread_layout=A_tile_layout](A_smem, A_tile)
            copy_dram_to_sram_async[thread_layout=B_tile_layout](B_smem, B_tile)

            # We need async_copy_wait_all to wait THIS thread_id finished its dram-to-sram
            # copying. It does not tell anything about the neighbor threads in this block.
            # Without async_copy_wait_all, because copy_dram_to_sram_async is non-blocking,
            # we will likely have all neighbor threads pass the barrier and the copy is not
            # done yet!
            async_copy_wait_all()

            # Here needs a barrier to ensures all threads have finished their chunk
            # of data to shared memory before any thread starts reading from it (RAW).
            barrier()

            if participates_in_compute:
                for k in range(BK):
                    var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                    var B_element = B_smem[k, subtile_col]
                    for t in range(TM):
                        product = A_subtile[t, 0] * B_element
                        dst_reg[t] += product

            # This barrier() ensures all threads have read shared memory before the
            # next-iteraion overwriting to shared memory (WAR).
            barrier()

        if participates_in_compute:
            dst_subtile.copy_from(dst_reg)


fn block_tiled_matrix_multiplication[
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

        var participates_in_compute = (
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

        var A_reg = LayoutTensor[
            dtype,
            Layout.row_major(TM),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        var B_reg = LayoutTensor[
            dtype,
            Layout.row_major(TN),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(COMPUTE_THREADS // BK, BK)
            comptime B_tile_layout = Layout.row_major(BK, COMPUTE_THREADS // BK)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            copy_dram_to_sram_async[thread_layout=A_tile_layout](A_smem, A_tile)
            copy_dram_to_sram_async[thread_layout=B_tile_layout](B_smem, B_tile)
            async_copy_wait_all()
            barrier()

            if participates_in_compute:
                for k in range(BK):
                    var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                    var B_subtile = B_smem.tile[1, TN](k, subtile_col)
                    A_reg.copy_from(A_subtile)
                    B_reg.copy_from(B_subtile)

                    outer_product_acc(dst_reg, A_reg, B_reg)

            barrier()

        if participates_in_compute:
            dst_subtile.copy_from(dst_reg)


fn block_tiled_matrix_multiplication_vectorized[
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
        alias simd_width = simd_width_of[dtype]()

        var subtile_row = Int(thread_idx.x // Int(BN // TN))
        var subtile_col = Int(thread_idx.x % Int(BN // TN))

        var participates_in_compute = (
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

        var dst_subtile_vec = dst_subtile.vectorize[1, simd_width]()
        var dst_reg_vec = dst_reg.vectorize[1, simd_width]()

        dst_reg_vec.copy_from(dst_subtile_vec)

        var A_reg = LayoutTensor[
            dtype,
            Layout.row_major(TM),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        var B_reg = LayoutTensor[
            dtype,
            Layout.row_major(TN),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(COMPUTE_THREADS // BK, BK)
            comptime B_tile_layout = Layout.row_major(BK, COMPUTE_THREADS // BK)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            copy_dram_to_sram_async[thread_layout=A_tile_layout](
                A_smem.vectorize[simd_width, 1](),
                A_tile.vectorize[simd_width, 1]()
            )
            copy_dram_to_sram_async[thread_layout=B_tile_layout](
                B_smem.vectorize[1, simd_width](),
                B_tile.vectorize[1, simd_width](),
            )
            async_copy_wait_all()
            barrier()

            if participates_in_compute:
                for k in range(BK):
                    var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                    var B_subtile = B_smem.tile[1, TN](k, subtile_col)
                    A_reg.copy_from(A_subtile)
                    B_reg.copy_from(B_subtile)

                    outer_product_acc(dst_reg, A_reg, B_reg)

            barrier()

        if participates_in_compute:
            dst_subtile_vec.copy_from(dst_reg_vec)


fn tensor_core_matmul_kernel[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, WM: Int, WN: Int, MMA_M: Int, MMA_K: Int, MMA_N: Int
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        warp_y = warp_id() // UInt(BN // WN)
        warp_x = warp_id() % UInt(BN // WN)

        var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x)
                         .tile[WM, WN](Int(warp_y), Int(warp_x))

        mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

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

            copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](A_smem_tile, A_dram_tile)
            copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](B_smem_tile, B_dram_tile)
            async_copy_wait_all()
            barrier()

            A_warp_tile = A_smem_tile.tile[WM, BK](Int(warp_y), 0)
            B_warp_tile = B_smem_tile.tile[BK, WN](0, Int(warp_x))

            for mma_k in range(BK // MMA_K):
                for mma_m in range(WM // MMA_M):
                    for mma_n in range(WN // MMA_N):
                        A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                        B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                        C_result = C_reg_tile.tile[1, 4](mma_m, mma_n)
                        var result = mma_op.mma_op(
                            mma_op.load_a(A_mma_tile),
                            mma_op.load_b(B_mma_tile),
                            C_result
                        )
                        C_result.copy_from(result)
            barrier()

        for mma_m in range(WM // MMA_M):
            for mma_n in range(WN // MMA_N):
                C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
                C_result = C_reg_tile.tile[1, 4](mma_m, mma_n)
                mma_op.store_d(C_mma_tile, C_result)


@compiler.register("my_matmul")
struct MyMatMul[algorithm: StaticString]:
    @staticmethod
    fn execute(
        raw_output: OutputTensor[rank=2],
        raw_A: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        raw_B: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        device_ctx = ctx.get_device_context()

        A = raw_A.to_layout_tensor()
        B = raw_B.to_layout_tensor()
        output = raw_output.to_layout_tensor()

        M = A.shape[0]()
        N = B.shape[1]()

        device_ctx.enqueue_memset(
            DeviceBuffer[output.dtype](
                device_ctx,
                output.ptr,
                (M * N),
                owning=False,
            ),
            0, # fill zeros
        )

        comptime OPTIMIZED_BLOCK_SIZE = 16 if has_amd_gpu_accelerator() else 32

        if algorithm == "naive":
            comptime BM = OPTIMIZED_BLOCK_SIZE
            comptime BN = OPTIMIZED_BLOCK_SIZE

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
            comptime BM = OPTIMIZED_BLOCK_SIZE
            comptime BN = OPTIMIZED_BLOCK_SIZE

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
            comptime BM = OPTIMIZED_BLOCK_SIZE
            comptime BN = OPTIMIZED_BLOCK_SIZE
            comptime BK = OPTIMIZED_BLOCK_SIZE

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
            comptime BM = 128
            comptime BN = 128
            comptime BK = 8

            comptime TM = 16
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

        elif algorithm == "block_tiling":
            comptime BM = 128
            comptime BN = 128
            comptime BK = 8

            comptime TM = 4
            comptime TN = 4
            comptime COMPUTE_THREADS = (BM * BN) // (TM * TN)
            comptime COPY_THREADS = max(BM * BK, BK * BN)
            comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)

            device_ctx.enqueue_function[
                block_tiled_matrix_multiplication[
                    output.dtype, A.layout, B.layout, output.layout,
                    BM, BK, BN, TM, TN, COMPUTE_THREADS
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=NUM_THREADS,
            )

        elif algorithm == "block_tiling_vectorized":
            comptime BM = 64
            comptime BN = 64
            comptime BK = 16

            comptime TM = 4
            comptime TN = 4
            comptime COMPUTE_THREADS = (BM * BN) // (TM * TN)
            comptime COPY_THREADS = max(BM * BK, BK * BN)
            comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)

            device_ctx.enqueue_function[
                block_tiled_matrix_multiplication_vectorized[
                    output.dtype, A.layout, B.layout, output.layout,
                    BM, BK, BN, TM, TN, COMPUTE_THREADS
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=NUM_THREADS,
            )

        elif algorithm == "tensor_core_matmul":
            comptime BM = 64
            comptime BN = 64
            comptime BK = OPTIMIZED_BLOCK_SIZE

            comptime WM = 32
            comptime WN = WARP_SIZE

            comptime MMA_M = 16
            comptime MMA_N = 16 if has_amd_gpu_accelerator() else 8
            comptime MMA_K = 4
            comptime NUM_WARPS = (BM // WM) * (BN // WN)

            device_ctx.enqueue_function[
                tensor_core_matmul_kernel[
                    output.dtype, A.layout, B.layout, output.layout,
                    BM, BK, BN,
                    WM, WN,
                    MMA_M, MMA_K, MMA_N
                ]
            ](
                A, B, output,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_WARPS * WARP_SIZE)
            )

        else:
            raise Error("Unknown algorithm:", algorithm)
