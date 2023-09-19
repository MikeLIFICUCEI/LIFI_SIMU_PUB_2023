# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:12:28 2023

@author: x
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

#sistema a resolver
def sistema(y,t,parametros):
    x1,v1,x2,v2=y
    m1,m2,k1,k2,b1,b2,L1,L2,mu1,mu2=parametros
    dydt=[v1,
          (-b1*v1-k1*(x1-L1)/m1+k2*(x2-x1-L2))/m1+mu1*(x1-L1)**3/m1+mu2*(x1-x2)**3/m1,
          v2,
          (-b2*v2-k2*(x2-x1-L2))/m2+mu2*(x2-x1-L2)**3]
    return dydt

#constantes
m1 = 1 
m2 = 1
k1 = 0.4
k2= 1.808
b1 = 0
b2 = 0
L1 = 0
L2 = 0
mu1= -1/6
mu2=-1/10
#parametros
parametros=[m1,m2,k1,k2,b1,b2,L1,L2,mu1,mu2]
#condiciones iniciales
y0=[1,0,-0.5,0]
#tiempo
t=np.linspace(0,50,1000)

sol=odeint(sistema,y0,t, args = (parametros,))
x1=sol[:,0]
v1=sol[:,1]
x2=sol[:,2]
v2=sol[:,3]

fig1 =  plt.figure(1, figsize=(10,10))
cx=fig1.add_subplot(212)
cx.plot(t,x1, color= 'blue', label= 'resorte 1')
cx.plot(t,x2, color= 'red', label= 'resorte 2')
cx.set_xlabel('Tiempo')
plt.tight_layout()
plt.show
