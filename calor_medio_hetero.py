# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:17:02 2023

@author: Jorge M
"""

# -*- coding: utf-8 -*-
"""
@author: jorge
"""

import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def solver(I, alpha, f, L, Nx, Nt,dx,dt, theta, u_L, u_R):
    """
    Theta = 0: Euler adelantado
    Theta = 1: Euler atrasado
    Theta = 1/2: Crank-Nicolson
    """
    x = np.linspace(0, L, Nx+1)
    T = Nt*dt
    t = np.linspace(0, T, Nt+1)
    print('dt=%g' % dt)
    print('dx=%g' % dx)
    
    if isinstance(alpha, (float,int)):
        alpha = np.zeros(Nx+1) + alpha
    elif callable(alpha):
        # calculamos alpha con la funcion
        a_0 = np.zeros(x.shape)
        for i in range(Nx+1):
            a_0[i] = alpha(x[i])
        alpha = a_0
    # calculamos mu con alpha    
    mu = alpha*(dt/dx**2)   

    # Condiciones de frontera constantes    
    if isinstance(u_L, (float,int)):
        u_L_ = float(u_L)  
        u_L = lambda t: u_L_
    if isinstance(u_R, (float,int)):
        u_R_ = float(u_R)
        u_R = lambda t: u_R_

    # Funciones para las CI y el forzamiento
    if f is None or f == 0:
        f = lambda x, t: 0
    elif isinstance(f, (float,int)):
        f_ = float(f)  
        f = lambda x, t: f_
    if I is None or I == 0:
        I = lambda x: 0    
        
    u   = np.zeros(Nx+1)   
    u_n = np.zeros(Nx+1)   

    mul = 0.5*mu*theta
    mur = 0.5*mu*(1-theta)

    diagonal = np.zeros(Nx+1)
    lower    = np.zeros(Nx)
    upper    = np.zeros(Nx)
    b        = np.zeros(Nx+1)
    # Generamos las diagonales
    diagonal[1:-1] = 1 + mul[1:-1]*(alpha[2:] + 2*alpha[1:-1] + alpha[:-2])
    lower[:-1] = -mul[1:-1]*(alpha[1:-1] + alpha[:-2])
    upper[1:]  = -mul[1:-1]*(alpha[2:] + alpha[1:-1])
    # condiciones de frontera
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0
    # Armamos la matriz dispersa
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx+1, Nx+1),
        format='csr')
    #Aplicamos condiciones iniciales
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])
    # generamos un espacio vacio para la solucion    
    sol = np.zeros((Nt,len(u)))
    # resolvemos para cada paso de tiempo
    for n in range(0, Nt):
        b[1:-1] = u_n[1:-1] + mur[1:-1]*(
            (alpha[2:] + alpha[1:-1])*(u_n[2:] - u_n[1:-1]) -
            (alpha[1:-1] + alpha[0:-2])*(u_n[1:-1] - u_n[:-2])) + \
            dt*theta*f(x[1:-1], t[n+1]) + \
            dt*(1-theta)*f(x[1:-1], t[n])
        # Condiciones de frontera
        b[0]  = u_L(t[n+1])
        b[-1] = b[-2]#u_R(t[n+1])
        u[:] = scipy.sparse.linalg.spsolve(A, b)
        u_n[:] = u[:]
        sol[n] = u[:]
        
    return u, x, t, sol, mu

#PARAMETROS

L = 1
I = 0 #condiciones iniciales
#Alphas
a0 =1
a1 = a0/2
a2 = a1/2

b= [0.25,0.5]  #distancias
f = 0 #forzamiento

Nx = 100
Nt = 200

dt = 0.01
dx = L/Nx

#CONDICIONES DE FRONTERA
u_L = 100
u_R = 0
#funcion alpha

x = np.linspace(0, L, Nx+1)
alpha =  np.piecewise(x, [x<b[0], ((x >= b[0]) & (x < b[1])),
                          ((x >= b[1]) & (x <= L))], 
                      [lambda x: a0, 
                       lambda x: a1,
                       lambda x: a2 ])

theta = 0.5 #metodo

u, x, t, sol, mu = solver(I, alpha, f, L, Nx, Nt,dx,dt, theta, u_L, u_R)
 
plt.contourf(x,t[:-1],sol, cmap = 'inferno')