import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
)


@compiler.register("my_matmul")
struct MyMatMul[algorithm: StaticString]:
    @staticmethod
    fn execute(
        output: OutputTensor[rank=2],
        A: InputTensor[dtype = output.dtype, rank = output.rank],
        B: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        print("algo: " + algorithm)
        #@parameter

