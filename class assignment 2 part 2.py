# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:17:44 2024

@author: shikh
"""

import torch 
import numpy as np

J = torch.FloatTensor(np.array([[1, -2, 1], [2, 3, -4], [1, -4, 1]])) # J must be in size 3 x 3
h = torch.FloatTensor(np.array([[1], [1], [1]]))# h must be in size 3 x 1
x0 = torch.FloatTensor(np.array([[0.1], [0.1], [0.1]])) # x0 must be in size 3 x 1
alpha = 1e-3
num_epochs = 10000
print (J. size () )
print (h. size () )
print ( x0 . size () )
x = x0
num_epochs = 10000
for epoch in range ( num_epochs ):
    print ('Epoch ', epoch )
    # Create a variable for x 
    
    x_var = torch . nn . Parameter (x , requires_grad = True )
    # Compute the objective L
    a = torch.matmul(J, x_var)
    b = torch.matmul(h.T, x_var)
    L = torch.matmul(x_var.T, a) + b
    # Compute the gradient by automatic d i f f e r e n t i a t i o n
    L.backward()
    # Gradient Descent update
    x = x - alpha * x_var.grad
    # Make sure that x is in range [ -1, 1]
    x[x < -1] = -1
    x[x > 1] = 1
print (' --------------------')
print ('Solution :', x)