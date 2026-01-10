from python import Python, PythonObject


@register_passable("trivial")
struct MockGPUIndex[idx_name: StaticString]:
    fn __init__(out self):
        pass

    @staticmethod
    fn get_pyobj() -> PythonObject:
        try:
            Python.add_to_path("./visualization_utils")
            return Python.import_module("mock_gpu")
        except e:
            print(e)
            return PythonObject()

    fn __getattr__[dim: StaticString](self) -> Int:
        try:
            var pyobj = Self.get_pyobj()
            return Int(pyobj.mock_primitives.get(idx_name, dim))
        except e:
            print(e)
            return 0

    @staticmethod
    fn set_dim3(x: Int, y: Int = 0, z: Int = 0):
        try:
            var pyobj = Self.get_pyobj()
            pyobj.mock_primitives.set(idx_name, 'x', x)
            pyobj.mock_primitives.set(idx_name, 'y', y)
            pyobj.mock_primitives.set(idx_name, 'z', z)
        except e:
            print(e)
            return


comptime block_idx = MockGPUIndex['block_idx']()
comptime thread_idx = MockGPUIndex['thread_idx']()


fn barrier():
    return


comptime WARP_SIZE = 8


fn warp_id() -> UInt:
    return UInt(thread_idx.x // WARP_SIZE)
