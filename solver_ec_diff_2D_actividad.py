# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:16:44 2023

@author: Jorge M
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from timeit import default_timer as timer

def solver_dif(I, a, f, Lx, Ly, Nx, Ny, dt, T, mask, IBV, theta=0.5,
    U_0x=0, U_0y=0, U_Lx=0, U_Ly=0):
    """
    theta = 0:   Euler adelantado
    theta = 1:   Implícito
    theta = 0.5: Crank-Nicolson (por defecto)
    """
    x = np.linspace(0, Lx, Nx+1)       
    y = np.linspace(0, Ly, Ny+1)       
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1) 

    Fx = a*dt/dx**2
    Fy = a*dt/dy**2

    # f(x,y,t)
    if f is None or f == 0:
        f = lambda x, y, t: np.zeros((x.size, y.size)) \
            if isinstance(x, np.ndarray) else 0

    u   = np.zeros((Nx+1, Ny+1))    
    u_n = np.zeros((Nx+1, Ny+1))    
    sol = np.zeros((Nx+1,Ny+1,Nt)) 

    Ix = list(range(0, Nx+1))
    Iy = list(range(0, Ny+1))
    It = list(range(0, Nt+1))

    # Para poder usar funciones en las fronteras
    if isinstance(U_0x, (float,int)):
        _U_0x = float(U_0x)  
        U_0x = lambda t: _U_0x
    if isinstance(U_0y, (float,int)):
        _U_0y = float(U_0y)  
        U_0y = lambda t: _U_0y
    if isinstance(U_Lx, (float,int)):
        _U_Lx = float(U_Lx)  
        U_Lx = lambda t: _U_Lx
    if isinstance(U_Ly, (float,int)):
        _U_Ly = float(U_Ly)  
        U_Ly = lambda t: _U_Ly

    # Condición inicial
    if isinstance(I, (float,int, np.ndarray)):
        u_n = I # Para poder dar una condición constante
    else:
        for i in Ix:
            for j in Iy:
                u_n[i,j] = I(x[i], y[j]) # o función
    u_n = np.where(mask.astype(bool), u_n, IBV)
                
    xv = x[:,np.newaxis]
    yv = y[np.newaxis,:]

    N = (Nx+1)*(Ny+1)
    main   = np.zeros(N)            # diagonal
    lower  = np.zeros(N-1)          # sub_diagonal
    upper  = np.zeros(N-1)          # super_diagonal
    lower2 = np.zeros(N-(Nx+1))     # diag abajo
    upper2 = np.zeros(N-(Nx+1))     # diag arriba
    b      = np.zeros(N)            # términos independientes

    # calculamos los elementos de la diagonal 
    lower_offset = 1
    lower2_offset = Nx+1

    m = lambda i, j: j*(Nx+1) + i
    j = 0; main[m(0,j):m(Nx+1,j)] = 1  
    for j in Iy[1:-1]:             
        i = 0;   main[m(i,j)] = 1 
        i = Nx;  main[m(i,j)] = 1  
        
        lower2[m(1,j)-lower2_offset:m(Nx,j)-lower2_offset] = - theta*Fy
        lower[m(1,j)-lower_offset:m(Nx,j)-lower_offset] = - theta*Fx
        main[m(1,j):m(Nx,j)] = 1 + 2*theta*(Fx+Fy)
        upper[m(1,j):m(Nx,j)] = - theta*Fx
        upper2[m(1,j):m(Nx,j)] = - theta*Fy
    j = Ny; main[m(0,j):m(Nx+1,j)] = 1  

    A = scipy.sparse.diags(
        diagonals=[main, lower, upper, lower2, upper2],
        offsets=[0, -lower_offset, lower_offset,
                 -lower2_offset, lower2_offset],
        shape=(N, N), format='csc')

    # Iteración en el tiempo
    for n in It[0:-1]:
        # Calculamos f
        f_a_np1 = f(xv, yv, t[n+1])
        f_a_n   = f(xv, yv, t[n])

        j = 0; b[m(0,j):m(Nx+1,j)] = U_0y(t[n+1])     
        for j in Iy[1:-1]:
            i = 0;   p = m(i,j);  b[p] = U_0x(t[n+1]) 
            i = Nx;  p = m(i,j);  b[p] = U_Lx(t[n+1]) 
            imin = Ix[1]
            imax = Ix[-1]  
            b[m(imin,j):m(imax,j)] = u_n[imin:imax,j] + \
                  (1-theta)*(Fx*(
              u_n[imin+1:imax+1,j] -
            2*u_n[imin:imax,j] +
              u_n[imin-1:imax-1,j]) +
                             Fy*(
              u_n[imin:imax,j+1] -
            2*u_n[imin:imax,j] +
              u_n[imin:imax,j-1])) + \
                theta*dt*f_a_np1[imin:imax,j] + \
              (1-theta)*dt*f_a_n[imin:imax,j]
        j = Ny;  b[m(0,j):m(Nx+1,j)] = U_Ly(t[n+1]) 
        # Resolvemos el sistema A*c = b
        c = scipy.sparse.linalg.spsolve(A, b)      
        # re mapeamos c en u
        u[:,:] = c.reshape(Ny+1,Nx+1).T
        u = np.where(mask.astype(bool), u, IBV)
         
        # Condiciones de frontera
        u[:,-1] = u[:,-2]
        u[:,0] = u[:,1]
        u[0,:] = u[1,:] 
        u[-1,:] = u[-2,:]
        # Actualizamos u_n antes del siguiente paso de tiempo
        u_n, u = u, u_n
        sol[:,:,n] = u
    return x, y, t, sol

I = 0 # condiciones iniciales
#Longitudes
Lx = 1 
Ly = 1 
#Tamano de matriz
Nx = 80 
Ny = 80
#Parametros
alpha = 0.01 #conductividad
f = 0 #forzamiento
IBV = 0 #no se pa que sirve
#Tiempo
dt = 0.05
T = 5
#condiciones de frontera
U_0y = 100 # lado izquierdo
U_0x = 0 # lado de abajo
U_Ly = 0 # lado derecho
U_Lx = 0 # lado de arriba


def mascara (Nx, Ny, i1, i2, j1,j2):#sirve para definir dominios
    m1 = int(i1*Ny)
    m2 = int(i2*Ny)
    n1 = int(j1*Nx)
    n2 = int(j2*Nx)
    mask = np.ones([Nx +1, Ny+1])
    mask[n1:n2,m1:m2] = 0;
    return mask
mask = mascara(Nx, Ny, 0.1, 0.15, 0.2,0.3)

theta=0.5
x, y, t, sol = solver_dif(I, alpha, f, Lx, Ly, Nx, Ny, dt, T, mask, IBV, theta,
    U_0x, U_0y, U_Lx, U_Ly)


def tiempos (dt,instante):
    N = int(instante/dt)
    return N
instantes= [0.1,0.5,1,4]
t_graf= np.zeros(len(instantes))
m,n,q = sol.shape
X, Y = np.meshgrid(x,y)

 
for j in range(len(instantes)):
    arg = 220+j+1
    t_graf[j]=int(tiempos(dt, instantes[j]))
    plt.figure(2)
    plt.subplot(arg)
    plt.pcolormesh(X,Y,sol[:,:,int(t_graf[j])],cmap = "inferno",shading = "auto")
    plt.suptitle("Conducción de calor en superfice plana",fontsize = 15)
    plt.text(0.65, 0.7, "tiempo ="+str(instantes[j]),color = "white",fontsize = 8)#+str(instantes[j-1]))
   

fig =plt.figure(1)
plts = []
ax = fig.add_subplot(111)
for i in range(q):
    pcol =ax.pcolormesh(X,Y, sol[:,:,i], cmap = 'inferno', shading = "auto")
    plts.append([pcol])
plt.colorbar(pcol)
ani = animation.ArtistAnimation(fig, plts, interval = 100, blit = True)
plt.show() 