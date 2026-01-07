struct MockGPUIndex:
    var x: Int
    var y: Int
    var z: Int

    fn __init__(out self, x: Int = 0, y: Int = 0, z: Int = 0):
        self.x = x
        self.y = y
        self.z = z


comptime block_idx = MockGPUIndex()
comptime thread_idx = MockGPUIndex()


fn barrier():
    return
