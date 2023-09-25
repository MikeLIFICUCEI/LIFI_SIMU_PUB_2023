#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:32:19 2023

@author: jair
"""
import numpy as np
import matplotlib.pyplot as plt


def euler_solv (Nt ,dx ,dt ,u0 ,L ,alpha):
    
    # Parametros de estabilidad:
    mu     = (dt/(dx**2))    
    # Dominio espacio-temporarl
    x      = np.arange(0, L+dx, dx)
    t      = np.arange(0, Nt+dt, dt)    
    sol    = np.zeros((Nt, len(x)))
    u_n    = np.zeros(len(x))    
    # Calculamos condiciones iniciales:
    u0     = u0(x)    
    # Aplicamos condiciones iniciales
    u      = u0    
    # Aplicamos condiciones de frontera:
    u[0]   = 100
    u[-1]  = 0    
    # Resolvemos
    for n in range(Nt):
        u_n[1:-1]   = u[1:-1] + mu*(u[:-2] - 2*u[1-1] + u[2:])
        # Asignamos a u_n a u
        u           = u_n
        # Aplicamos condiciones de frontera para el siguiente paso:
        u[0]        = 100
        u[-1]       = 0        
        # Aplicamos condiciones de frontera 
        sol[n]      = u
    return x, t, mu, sol
        


#Planteamos los parametros del problema en particular:

L = 1 
Nt = 500
dx = 0.05
dt = 0.0004
alpha = 1

sigma = 0.5
u0 = lambda x: np.exp(-1/2*((x-L/2)**2/(sigma**2)))

x,t,mu,sol = euler_solv(Nt, dx, dt, u0, L, alpha)

plt.figure(1)
plt.contourf(x,t,sol)
plt.ylabel('Tiempo[s]')
plt.xlabel('Longitud[m]')
plt.colorbar()



