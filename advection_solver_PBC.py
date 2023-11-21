# -*- coding: utf-8 -*-
"""
@author: Jorge M. Montes A. 

Curso:  Simulación de Procesos Físicos
        Licenciatura en Física
        CUCEI, UdeG.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solver(I, U0, v, L, dt, C, T,esquema, cf_periodicas=True):
    """
    esquema : String,
        FE: Esquema forwad Euler
        LF: Esquema Leap Frog
        UP: Esquema Upwind
        LW: Esquema Lax-Wendroff
        
    cf_periodicas : Bolean, opcional
        True: condiciones de frontera periódicas
        False: Fronteras abiertas
    """
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # puntos de malla en tiempo
    dx = v*dt/C
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # puntos de malla en espacio
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    C = v*dt/dx
    # algo de información para checar el valor de C
    print('dt=%g, dx=%g, Nx=%d, C=%g' %(dt, dx, Nx, C))

    u   = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)
    u_nm1 = np.zeros(Nx+1)
    sol = np.zeros((Nt+1,Nx+1))

    # Condiciones iniciales u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    # Imponemos condiciones de frontera
    u[0] = U0

    for n in range(0, Nt):
        if esquema == 'FE': # Esquema Forwad Euler
            if cf_periodicas:
                i = 0
                u[i] = u_n[i] - 0.5*C*(u_n[i+1] - u_n[Nx])
                u[Nx] = u[0]
                #u[i] = u_n[i] - 0.5*C*(u_n[1] - u_n[Nx])
            for i in range(1, Nx):
                u[i] = u_n[i] - 0.5*C*(u_n[i+1] - u_n[i-1])
        elif esquema == 'LF': # Esquema Leap Frog
            if n == 0:
                # Usamos upwind para la primera iteración
                if cf_periodicas:
                    i = 0
                    #u[i] = u_n[i] - C*(u_n[i] - u_n[Nx-1])
                    u_n[i] = u_n[Nx]
                for i in range(1, Nx+1):
                    u[i] = u_n[i] - C*(u_n[i] - u_n[i-1])
            else: # para los siguientes pasos de tiempo
                if cf_periodicas:
                    i = 0
                    u[i] = u_nm1[i] - C*(u_n[i+1] - u_n[Nx-1])
                    #u_n[i] = u_n[Nx]
                for i in range(1, Nx):
                    u[i] = u_nm1[i] - C*(u_n[i+1] - u_n[i-1])
                if cf_periodicas:
                    u[Nx] = u[0]
        elif esquema == 'UP': # Esquema Upwind
            if cf_periodicas:
                u_n[0] = u_n[Nx]
            for i in range(1, Nx+1):
                u[i] = u_n[i] - C*(u_n[i] - u_n[i-1])
        elif esquema == 'LW': # Esquema Lax-Wendroff
            if cf_periodicas:
                i = 0
                u[i] = u_n[i] - 0.5*C*(u_n[i+1] - u_n[Nx-1]) + \
                       0.5*C*(u_n[i+1] - 2*u_n[i] + u_n[Nx-1])
                #u_n[i] = u_n[Nx]
            for i in range(1, Nx):
                u[i] = u_n[i] - 0.5*C*(u_n[i+1] - u_n[i-1]) + \
                       0.5*C*(u_n[i+1] - 2*u_n[i] + u_n[i-1])
            if cf_periodicas:
                u[Nx] = u[0]
        else:
            raise ValueError('Esquema ="%s" no implemantado' % esquema)

        if not cf_periodicas:
            # imponemos condiciones de frontera
            u[0] = U0

        # Cambiamos las variables antes del siguiente paso de tiempo
        u_nm1, u_n, u = u_n, u, u_nm1
        sol[n] = u
    return u, x, t, sol
