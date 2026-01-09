import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList


@compiler.register("my_add_constant")
struct MyAddConstant[value: Int]:
    @staticmethod
    fn execute[target: StaticString](
        outp: OutputTensor,
        x: InputTensor[dtype = outp.dtype, rank = outp.rank],
        ctx: DeviceContextPtr,
    ) raises:

        @parameter
        @always_inline
        fn add_constant[width: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + value

        foreach[add_constant, target=target](outp, ctx)
