import torch

a = torch.rand((100, 100), device='cuda')
b = torch.rand((100, 100), device='cuda')

def fn(x, y):
    z = torch.matmul(x, y)
    return torch.nn.functional.softmax(z, dim=1)

compiled_fn = torch.compile(fn)
print(compiled_fn(a, b))
