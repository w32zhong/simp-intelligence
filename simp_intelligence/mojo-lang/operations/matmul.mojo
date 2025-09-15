import compiler
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


fn naive_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout
    ](
        A: LayoutTensor[dtype, A_layout, MutableAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutableAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutableAnyOrigin],
    ):
    pass


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
        alias OPTIMIZED_BLOCK_SIZE = 32 if has_nvidia_gpu_accelerator() else 16

        A = raw_A.to_layout_tensor()
        B = raw_B.to_layout_tensor()
        output = raw_output.to_layout_tensor()

        M = A.shape[0]()
        N = B.shape[1]()

        alias BM = OPTIMIZED_BLOCK_SIZE
        alias BN = OPTIMIZED_BLOCK_SIZE

        @parameter
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
