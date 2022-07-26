import torch

# ======================================================= #
#                    Initializing Tensor                  #
# ======================================================= #

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5, 5)  # I
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)  # tensor([0, 1, 2, 3])
print(tensor.bool())
print(tensor.short())
print(tensor.long())  # *** int64 ***
print(tensor.half())  # float16
print(tensor.flaot())  # *** float32 ***
print(tensor.double())  # float64

# Array to tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
