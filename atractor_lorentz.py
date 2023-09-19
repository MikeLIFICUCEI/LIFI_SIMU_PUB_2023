# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:09:28 2023

@author: x
"""

## ATRACTOR DE LORENTS ####
"""
x'=s(y-x)
y'=rx-y-xz
z'=xy-bz

r:num de rayleigh
s: nume de Prandtl
b: parametro asociado al tamaño fisico del sistema 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp


def atractor(v,t,parametros):
    x,y,z=v
    sigma,b,r = parametros
    dvdt=[sigma*(y-x),
          r*x-y-x*z,
          x*y-b*z]
    return dvdt
#PARAMETROS
sigma=10
b=8/3
r=28
parametros=[sigma,b,r]
#condiciones
p1=[0,2,0]  #sistema 1
p2=[0,2.000001,0] # sistema 2
#tiempo
t=np.linspace(0,50,5000)
#Soluciones
sol_1=odeint(atractor, p1, t, args=(parametros,))
sol_2=odeint(atractor, p2, t, args=(parametros,))
#sistema 1
x1 = sol_1[:,0]
y1 = sol_1[:,1]
z1 = sol_1[:,2]
#sistema 2
x2 = sol_2[:,0]
y2 = sol_2[:,1]
z2 = sol_2[:,2]

fig1 =  plt.figure(1, figsize=(10,10))
ax = fig1.add_subplot(221, projection ='3d')
ax.plot(x1,y1,z1)

bx = fig1.add_subplot(222, projection ='3d')
bx.plot(x2,y2,z2,color='red')

cx=fig1.add_subplot(212)
cx.plot(t,x1, color= 'blue', label= 'atractor 1')
cx.plot(t,x2, color= 'red', label= 'atractor 1')
cx.set_xlabel('Tiempo')
plt.tight_layout()
plt.show


