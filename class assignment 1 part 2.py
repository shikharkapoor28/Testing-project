# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:02:12 2024

@author: shikh
"""

import torch
def R(z): # z = (x, y)
    return torch.sum(z * z * z * z)  # x^2 + y^2

# For example: If z = (a, b, c) then R(z) = a^2 + b^2 + c^2

def grad_R(z):
    return 4*z**3

# Gradient Descent
z = torch.FloatTensor([-3, 2])
print('Iteration 0: z =', z)

alpha = 1
tol = 1e-6

count = 0
while True:
    z_next = z - alpha * grad_R(z)
    count += 1
    print('Iteration', count,': z =', z_next)
    if torch.norm(z - z_next) < tol:
        break
    z = z_next
