# A minimal code to demonstrate the core idea of backward(),
# taking the inspriation from both Pytorch and lucasdelimanogueira/PyNorch.
# Note that "broadcasting" and many other details are not handled in this toy.


import numpy as np
from collections import defaultdict
from contextlib import contextmanager


_grad_enabled = True


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc_value, traceback):
        global _grad_enabled
        _grad_enabled = self.prev


class AddBackward:
    def __init__(self, x, y):
        # Note: PyTorch uses a context object (ctx) to save tensors for backward, i.e.,
        # ctx.save_for_backward(x). Here we just store as Backward "input" for simplicity,
        # of course, this only works if backward() always happens in the same context.
        self.input = [x, y]
        self.saved_versions = [x._version, y._version]

    def backward(self, gradient):
        for t, v in zip(self.input, self.saved_versions):
            if t._version != v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an in-place operation")
        return [gradient, gradient]


class ScalarMulBackward:
    def __init__(self, x, scalar):
        self.input = [x]
        self.scalar = scalar
        self.saved_versions = [x._version]

    def backward(self, gradient):
        for t, v in zip(self.input, self.saved_versions):
            if t._version != v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an in-place operation")
        return [gradient * self.scalar]


class SumBackward:
    def __init__(self, x, axis=None):
        self.input = [x]
        self.axis = axis
        self.saved_versions = [x._version]

    def backward(self, gradient):
        for t, v in zip(self.input, self.saved_versions):
            if t._version != v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an in-place operation")

        if self.axis is None:
            # If axis is None, sum reduces the tensor to a scalar.
            grad_output = Tensor(gradient.data * np.ones_like(self.input[0].data))
        else:
            raise NotImplementedError
        return [grad_output]


class Tensor:
    def __init__(self, data=None, requires_grad=False, grad_fn=None):
        self.requires_grad = requires_grad
        self.retains_grad = False
        self.grad = None
        self.grad_fn = grad_fn if requires_grad else None
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self._version = 0

    def is_non_leaf(self):
        return self.grad_fn is not None

    def retain_grad(self):
        self.retains_grad = True

    def _accum_grads(self, grad):
        print(self, 'accum_grads:', grad)
        if self.grad is None:
            self.grad = 0
        self.grad += grad.data

    def __repr__(self):
        repr = f'Tensor({self.data}, '
        if self.requires_grad:
            repr += f'requires_grad={self.requires_grad}, '
        if self.grad_fn is not None:
            repr += f'grad_fn={self.grad_fn.__class__.__name__}, '
        if self.grad is not None:
            repr += f'grad={self.grad}, '
        return repr.rstrip(' ,') + ')'

    def __iadd__(self, other):
        self.data += other.data if isinstance(other, Tensor) else other
        self._version += 1
        return self

    def __add__(self, other):
        should_track = (self.requires_grad or other.requires_grad) and _grad_enabled
        return Tensor(
            self.data + other.data,
            requires_grad=should_track,
            grad_fn=AddBackward(self, other) if should_track else None
        )

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError
        should_track = self.requires_grad and _grad_enabled
        return Tensor(
            self.data * other,
            requires_grad=should_track,
            grad_fn=ScalarMulBackward(self, other) if should_track else None
        )

    def sum(self, axis=None):
        should_track = self.requires_grad and _grad_enabled
        return Tensor(
            self.data.sum(axis=axis),
            requires_grad=should_track,
            grad_fn = SumBackward(self, axis) if should_track else None
        )

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def backward(self, gradient=None):
        if not self.requires_grad:
            # requires_grad decides where the backward pass stops
            return
        elif gradient is not None and gradient.data.ndim != 0:
            # this toy code only handles a scalar gradient, in reality, pytorch allows
            # calling .backward() on non-scalar tensors using a Vector-Jacobian product.
            raise NotImplementedError

        # topological ordering. Also see Andrej Karpathy's micro_grad:
        # https://youtu.be/VMj-3S1tku0?si=T6W-N0vhXpoqQZIU&t=4692
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.is_non_leaf():
                    for u in v.grad_fn.input:
                        build_topo(u)
                topo.append(v)
        build_topo(self)

        # backprop and accumulate gradients
        gradient_dict = defaultdict(list)
        gradient_dict[self] = [gradient or Tensor(1)]
        for v in reversed(topo):
            # in a diamond graph, a vertex can have multiple incoming gradients
            gradient = Tensor(0)
            for grad in gradient_dict.get(v):
                gradient = Tensor(gradient.data + grad.data)

            # pytorch convention requires a non-leaf to explicitly retain gradients
            if not v.is_non_leaf() or v.retains_grad:
                v._accum_grads(gradient)

            # backprop to children ...
            if v.is_non_leaf():
                inputs = v.grad_fn.input
                input_grads = v.grad_fn.backward(gradient)
                for u, grad in zip(inputs, input_grads):
                    gradient_dict[u].append(grad)


def test_backward(t1):
    #         x2         x3
    # tensor1 -- tensor2 -- tensor3 -+ tensor5
    #       \__x4___ tensor4 _______/
    t2 = t1 * 2
    t2.retain_grad()
    t3 = t2 * 3
    t3.retain_grad()
    t4 = t1 * 4
    t4.retain_grad()
    t5 = (t3 + t4).sum()
    t5.retain_grad()
    print('t5', t5)
    print('t4', t4)
    print('t3', t3)
    print('t2', t2)
    print('t1', t1)
    t5.backward()
    print('t5.grad', t5.grad)
    print('t4.grad', t4.grad)
    print('t3.grad', t3.grad)
    print('t2.grad', t2.grad)
    print('t1.grad', t1.grad)
    print('-' * 80)


def test_backward_with_inplace_check():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = x * 2
    # Modify x in-place. This should increment x._version
    x += 1.0
    try:
        breakpoint()
        y.backward()
    except RuntimeError:
        print('in-place modification detected')


def test_no_grad_context():
    print("Testing no_grad context:")
    x = Tensor([1.0], requires_grad=True)
    with no_grad():
        y = x * 2

    if y.requires_grad or y.grad_fn is not None:
        print("FAILURE: y should not require grad inside no_grad block")
    else:
        print("SUCCESS: y does not require grad inside no_grad block")

    # Verify outside context
    z = x * 2
    if z.requires_grad and z.grad_fn is not None:
        print("SUCCESS: z requires grad outside no_grad block")
    else:
        print("FAILURE: z should require grad outside no_grad block")
    print('-' * 80)


def test_detach():
    print("Testing detach:")
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 3
    z = y.detach()

    if z.requires_grad or z.grad_fn is not None:
        print("FAILURE: detached tensor should not require grad")
    else:
        print("SUCCESS: detach works as expected")
    print('-' * 80)


if __name__ == "__main__":
    #t1 = Tensor([1, 2.1], requires_grad=True)
    #test_backward(t1)
    #import torch
    #t1 = torch.tensor([1, 2.1], requires_grad=True)
    #test_backward(t1)

    test_backward_with_inplace_check()

    #test_no_grad_context()

    #test_detach()
