# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:18:12 2024

@author: shikh
"""

import torch

n = 4

j = torch.randn(n, n)
h = torch.randn(n)
x = torch.randn(n)

alpha = 1e-3

x.requires_grad_(True)

a = x.T
b = torch.matmul(j,x)
c = torch.matmul(a,b)
d = torch.matmul(h.T,x)

l = c + d

print (l)

e = x.T 
f = j + j.T
g = torch.matmul(e,f)

grad_l = g + h.T

print (grad_l)


l.backward()

print (x.grad)

var = torch.norm(x.grad - grad_l)

if var < 1e-4:
    
    print ('correct formula')
else:
        print('Incorrect formula')