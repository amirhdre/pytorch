import torch

x = torch.arange(9)

x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)

# View and reshape difference
y = x_3x3.t()
print(y.contiguous().view(9))

#########
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

x1_flat = x1.flatten()
x1_flat = x1.view(-1)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

# keep dim 0 at 0 (old at new) & 2 at 1 & 1 at 2
z = x.permute(0, 2, 1)

# Transpose using permute: transpose is special case of permute
x = torch.arange(9).reshape(3, 3)
print(x.permute(1, 0))

x = torch.arange(10)
print(x.shape)  # [10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
z = x.squeeze(1)
print(z.shape)
