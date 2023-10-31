# -*- coding: utf-8 -*-
"""
Universidad de Guadalajara
CUCEI, Licenciatura en Física
Curso: Simulación de Procesos Físicos.
@author: Jorge M. Montes Arechiga, Departamento de Física.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def ec_onda(I, V, f, c, U_0, U_L, L, dt, C, T,beta):
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)      # Puntos de malla en tiempo

    # Encotramos max(c) para adaptar  dx a C y dt
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 101)])
        
    dx = dt*c_max/(beta*C) # beta es el factor de seguridad (beta=<1)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)     # Puntos de malla en espacio     
    # Obtenemos los dt y dx
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Para calcular c(x)
    if isinstance(c, (float,int)): # Si es un valor
        c = np.zeros(x.shape) + c
    elif callable(c):
        c = c(x)

    q = c**2
    C2 = (dt/dx)**2; dt2 = dt*dt    # variables de apoyo

    # Funciones para los argumentos de entrada 0 o None
    if f is None or f == 0:
        f = (lambda x, t: 0)
    if I is None or I == 0:
        I = (lambda x: 0)
    if V is None or V == 0:
        V = (lambda x: 0) 
    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0

    # Preasignamos espacio para las soluciones
    u     = np.zeros(Nx+1)   # Solucion
    u_n   = np.zeros(Nx+1)   # Solucion a 1 paso de tiempo atras
    u_nm1 = np.zeros(Nx+1)   # Solucion a 2 pasos de tiempo atras

    # Para la notacion de indices
    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Cargamos condiciones iniciales
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    # Formula para la primera iteracion
    for i in Ix[1:-1]:
        u[i] = u_n[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_n[i+1] - u_n[i]) - \
                0.5*(q[i] + q[i-1])*(u_n[i] - u_n[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    i = Ix[0]
    if U_0 is None:
        # Condiciones de frontera de la primera iteracion
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        u[i] = u_n[i] + dt*V(x[i]) + \
               0.5*C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
        0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_0(dt)

    i = Ix[-1]
    if U_L is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        u[i] = u_n[i] + dt*V(x[i]) + \
               0.5*C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
        0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_L(dt)

    # Actualizamos valores
    u_nm1, u_n, u = u_n, u, u_nm1
    
    # Preasignamos espacio para la solucion
    sol = np.zeros((Nt+1,Nx+1))
    # iteracion principal
    for n in It[1:-1]:
        for i in Ix[1:-1]:
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                C2*(0.5*(q[i] + q[i+1])*(u_n[i+1] - u_n[i])  - \
                    0.5*(q[i] + q[i-1])*(u_n[i] - u_n[i-1])) + \
                dt2*f(x[i], t[n])

    # Condiciones de frontera
        i = Ix[0]
        if U_0 is None:
            ip1 = i+1
            im1 = ip1
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
            dt2*f(x[i], t[n])
        else:
            u[i] = U_0(t[n+1])

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
            dt2*f(x[i], t[n])
        else:
             u[i] = U_L(t[n+1])

        # actualizamos los valores para la siguiente iteracion
        u_nm1, u_n, u = u_n, u, u_nm1
        sol[n] = u
    return u, x, t, sol

