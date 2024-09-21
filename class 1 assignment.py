# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:50:46 2024

@author: shikh
"""
import matplotlib.pyplot as image
import numpy as np

def eq(x):
    return x**3 + x**2

x = np.linspace(-5, 5, 100)
y=eq(x)

#create the diagram
image.figure(figsize=(4,1))
image.plot(x,y, label='y = x^3 +x^2', color='blue')
image.xlabel('x')
image.xlabel('y')
image.legend()

image.grid(True)

image.show()