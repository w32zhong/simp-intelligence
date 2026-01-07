class MockDim3:
    def __init__(self, x, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class MockPrimitives:
    def __init__(self):
        self.thread_idx = MockDim3(0)
        self.block_idx = MockDim3(0)

    def get(self, attr1, attr2):
        idx = getattr(self, attr1)
        return getattr(idx, attr2)

    def set(self, attr1, attr2, val):
        idx = getattr(self, attr1)
        setattr(idx, attr2, val)


mock_primitives = MockPrimitives()
