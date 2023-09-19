# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### INTEGRACION NUMERICA###

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quadrature 


def integral(a,b,N,f,metod):
    
    h=(b-a)/N
    if metod=="t":
        trapecio=(h/2)*np.sum(f[:-1]+f[1:])
        return trapecio
    if metod=="s":
        simpson=(h/3)*np.sum(f[0:-1:2]+4*f[1::2]+f[2::2])
        return simpson
    
a=0
b=2*np.pi
N=10
x=np.arange(a,b,N+1 )
fint=lambda x: np.sin(x)
fx=fint(x)
int_1= integral(a,b,N,fx,"t")
int_2=integral(a,b,N,fx,"s")

    
   
    
    