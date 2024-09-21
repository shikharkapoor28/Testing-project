# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:26:20 2024

@author: shikh
"""
import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

f = x**2 * y + y**2 * z + z**2 + 4

f.backward()

print("Function value f(x, y, z) = {:.1f}".format(f.item()))

print("df/dx: ", x.grad.item())

print("df/dy: ", y.grad.item())

print("df/dz: ", z.grad.item())