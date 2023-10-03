# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:42:29 2023

@author: x
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

# CONDUCCION DE CALOR#
#Método de Euler adelantado#
def euler(Nt,dx,dt,u0,L,alpha,uf_1,uf_2):
    #parametros de estabilidad
    mu = (dt/(dx**2))
    x = np.arange(0, L+dx, dx)
    t = np.arange(0, Nt*dt, dt)
    sol = np.zeros((Nt,len(x)))
    u_n = np.zeros(len(x))
    #calculo de C I
    u0 = u0(x)
    #aplicamos C I
    u = u0
    # condiciones de frontera
    u[0] = uf_1
    u[-1] = uf_2
    
    
    for n in range(Nt):
        u[0] = uf_1
        u[-1] = uf_2
        u_n[1:-1] = u[1:-1] + mu*(u[:-2] - 2*u[1:-1] + u[2:])
        u = u_n
        sol[n] = u
    return x, t, mu, sol
#PARAMETROS DEL PROBLEMA:
alpha = 1
L = 1
sigma = 0.5
Nt = 500
dx = 0.05
dt = 0.0004
#CONDICION INICIAL
u0 = lambda x:np. zeros (len (x))
#Condiciones de frontera
T_1 = 100
T_2 = 0


###
x,t,mu,sol = euler(Nt, dx, dt, u0, L, alpha, T_1, T_2)

plt.figure(1, figsize = [6,5])
plt.contourf(x,t,sol)
plt.ylabel('Tiempo[s]', fontsize = 10)
plt.xlabel('Longitud[m]' , fontsize = 10)

plt.suptitle("Conducción de calor en barra delgada",fontsize = 15)
plt.colorbar()

    
