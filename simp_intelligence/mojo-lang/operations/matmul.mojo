import compiler
from gpu import (
    block_dim,
    block_idx,
    thread_idx,
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
)
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
from sys.info import has_nvidia_gpu_accelerator
from utils import StaticTuple


#@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn naive_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        var row = block_dim.x * block_idx.x + thread_idx.x
        var col = block_dim.y * block_idx.y + thread_idx.y

        var dst_reg: C.element_type = 0

        if row < M and col < N:
            for k in range(K):
                dst_reg = dst_reg + A[row, k] * B[k, col]

        if row == 1 and col == 2:
            print(row, col, dst_reg)
            C[1, 2] = 1234
        else:
            C[row, col] = dst_reg


@compiler.register("my_matmul")
struct MyMatMul[algorithm: StaticString]:
    @staticmethod
    fn execute(
        raw_output: OutputTensor[rank=2],
        raw_A: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        raw_B: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        print("algo: " + algorithm)
        device_ctx = ctx.get_device_context()

        A = raw_A.to_layout_tensor()
        B = raw_B.to_layout_tensor()
        output = raw_output.to_layout_tensor()

        M = A.shape[0]()
        N = B.shape[1]()

        alias OPTIMIZED_BLOCK_SIZE = 16
        alias BM = OPTIMIZED_BLOCK_SIZE
        alias BN = OPTIMIZED_BLOCK_SIZE

        if algorithm == "naive":
            device_ctx.enqueue_function[
                naive_matmul[
                    output.dtype, A.layout, B.layout, output.layout
                ]
            ](
                A, B, output,
                grid_dim=(N // BN, M // BM),
                block_dim=(BN, BM),
            )
        else:
            raise Error("Unknown algorithm:", algorithm)

        device_ctx.synchronize()
