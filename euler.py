# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:05:30 2023

@author: x
"""

## METODO DE EULER###
""" Metodo de solucion de EDOs
yn+1 = yn + f(yn, tn)(tn+1 − tn)
solucionar y'=-ycos(t),  0:t:5"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

def euler(f,y0,t):
    y=np.empty(len(t)) #para generar un vector vacio
    y[0]=y0
    for n in range(0,len(t)-1):
        h = (t[n+1]-t[n])
        y[n+1] = y[n] + f(y[n],t[n])*h
    return y

def f(y,t):
    dydt=-y*np.cos(t)
    return dydt
y0=1/2
delta=0.1
t = np.arange(0,5+delta,delta )
ye = euler(f,y0,t) #sol numerica
ya =1/2*np.exp(-np.sin(t)) #sol 
yo=odeint(f,y0,t)

#graficas

fig, ax = plt.subplots()
ax.plot(t,ye, label = "euler")
ax.plot(t,ya, label = "analitica")
ax.plot(f,t,ya,label ="ode")
plt.legend(loc = 'upper right')
plt.show()
