# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:19:41 2023

@author: x
"""
#metodo de Euler atrasado
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

def back_euler(u0, L, Nx, dx, Nt, dt, alpha):
    x = np.arange(0, L, dx)
    t = np.arange(0, Nt*dt, dt)
    mu = alpha*dt/(dx**2)
    
    #espacios vacios
    u_n = np.zeros(Nx)
    u_nm1 = np.zeros(Nx)
    sol = np.zeros((Nt,len(x)))
    
    #creacion de las diagonales
    diag = np.zeros(Nx)
    infe = np.zeros(Nx-1)
    supe = np.zeros(Nx-1)
    b = np.zeros(Nx)
    
    diag[:] = 1+ 2*mu
    infe[:] = -mu
    supe[:] = -mu
    
    #condiciones de frontera en la matriz
    diag[0] = 1
    infe[-1] = 0
    supe[0] = 0
    diag[Nx-1] = 1
    
    A = scipy.sparse.diags(diagonals=[supe, diag, infe], offsets=[1,0,-1], 
                           shape=(Nx,Nx), format='csr')
    
    #aplicamos condiciones iniciales
    
    u0 = u0(x)
    u_nm1 = u0
    
    for n in range(0, Nt):
        b = u_nm1
        #condicion defrontera en u_nm1
        b[0] = 0
        b[-1] = 0
        #resolvemos el sistema
        u_n[:] = scipy.sparse.linalg.spsolve(A, b)
        sol[n] = u_n
        u_nm1 = u_n
    return x,t,mu,sol

alpha = 1
L = 1
Nx = 50
dx = L/Nx

Nt = 1000
dt = 0.01

sigma = 0.5
u0 = lambda x: np.exp(-1/2*((x-L/2)*2/(sigma*2)))



x,t,mu,sol = back_euler(u0, L, Nx, dx, Nt, dt, alpha)
plt.figure(1)
plt.contourf(x,t,sol)
plt.ylabel('Tiempo[s]')
plt.xlabel('Longitud[m]')
plt.colorbar()