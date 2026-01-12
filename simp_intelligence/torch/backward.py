# A minimal code to demonstrate the idea of backward()
# taking the inspriation from both Pytorch and lucasdelimanogueira/PyNorch>


import numpy as np


class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]


class ScalarMulBackward:
    def __init__(self, x, scalar):
        self.input = [x]
        self.scalar = scalar

    def backward(self, gradient):
        return [gradient * self.scalar]


class SumBackward:
    def __init__(self, x, axis=None):
        self.input = [x]
        self.axis = axis

    def backward(self, gradient):
        if self.axis is None:
            # If axis is None, sum reduces the tensor to a scalar.
            grad_output = Tensor(gradient.data * np.ones_like(self.input[0].data))
        else:
            raise NotImplementedError
        
        return [grad_output]


class Tensor:
    def __init__(self, data=None, requires_grad=False, grad_fn=None):
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn if requires_grad else None
        self.data = data if isinstance(data, np.ndarray) else np.array(data)

    def __repr__(self):
        repr = f'Tensor({self.data}, '
        if self.requires_grad:
            repr += f'requires_grad={self.requires_grad}, '
        if self.grad_fn is not None:
            repr += f'grad_fn={self.grad_fn.__class__.__name__}, '
        if self.grad is not None:
            repr += f'grad_fn={self.grad}, '
        return repr.rstrip(' ,') + ')'

    def __add__(self, other):
        return Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=AddBackward(self, other)
        )

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError
        return Tensor(
            self.data * other,
            requires_grad=self.requires_grad,
            grad_fn=ScalarMulBackward(self, other)
        )

    def sum(self, axis=None):
        return Tensor(
            self.data.sum(axis=axis),
            requires_grad=self.requires_grad,
            grad_fn = SumBackward(self, axis)
        )

    def is_leaf(self):
        return self.grad_fn is None

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
                
        if gradient is None:
            if self.data.ndim == 0:
                gradient = Tensor(1)
            else:
                raise RuntimeError("Gradient argument must be specified for non-scalar tensors.")

        # topological ordering. Also see Andrej Karpathy's micro_grad:
        # https://youtu.be/VMj-3S1tku0?si=T6W-N0vhXpoqQZIU&t=4692
        topo = []
        visited = set()
        def build_topo(v, grad):
            if v not in visited:
                visited.add(v)
                if not v.is_leaf():
                    inputs = v.grad_fn.input
                    input_grads = v.grad_fn.backward(grad)
                    for input_tensor, input_grad in zip(inputs, input_grads):
                        build_topo(input_tensor, input_grad)
                topo.append((v, grad))
        build_topo(self, gradient)

        for v, grad in reversed(topo):
            if v.grad is None:
                v.grad = 0
            v.grad += grad.data


if __name__ == "__main__":
    t1 = Tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True)
    t2 = Tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True)
    t3 = (t1 * 123 + t2 * 321).sum()
    print(t3)
    t3.backward()
    print(t3.grad)
    print(t1.grad)
    print(t2.grad)
    print('-' * 80)

    import torch
    v1 = torch.tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True)
    v2 = torch.tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True)
    v3 = (v1 * 123 + v2 * 321).sum()
    v3.retain_grad() # pytorch requires non-leaf root to explicitly call this to retain gradients.
    print(v3)
    v3.backward()
    print(v3.grad)
    print(v1.grad)
    print(v2.grad)
