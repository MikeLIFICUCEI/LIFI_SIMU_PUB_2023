# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:03:43 2023

@author: x
"""

###PENDULO SIMPLE
""""   O''(t)+BO'(t)-mgsin(O)"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

def pendulo(y,t,alpha,beta):
    theta,w= y
    dydt=[w, 
          -beta*w-alpha*np.sin(theta)]
    return dydt

#PARAMETROS CONSTANTES
g = 9.81
m = 1
L=1
b=0
#COEFICIENTES
beta=b/m
alpha=g/L
#TIEMPO
t_final = 30
t = np.linspace(0, t_final, t_final*10)
# CONDICIONES INICIALES
y0 = [np.pi/2,0]

#solucion
sol = odeint(pendulo,y0,t,args=(alpha,beta))

theta = sol[:,0] #toma la primer columna
w = sol[:,1]#toma la segunda columna

#graficas
fig = plt.subplots()
plt.plot(t,theta)
#ax.plot(t,w)

plt.show()


