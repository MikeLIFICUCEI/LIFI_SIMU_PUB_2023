# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:30:47 2023

@author: x
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, mask, IBV):
    x = np.linspace(0, Lx, Nx+1)  # Puntos de malla en x
    y = np.linspace(0, Ly, Ny+1)  # Puntos de malla en y
    xv = x[:,np.newaxis]          # Añadimos otra dimensión a los vectores
    yv = y[np.newaxis,:]
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)    # puntos de malla en tiempo
    # calculamos dt, dx, dy.
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]
    Cx2 = (c*dt/dx)**2;  Cy2 = (c*dt/dy)**2
    dt2 = dt**2
    
    if f is None or f == 0:
        f = lambda x, y, t: np.zeros((x.shape[0], y.shape[1]))
    if V is None or V == 0:
        V = lambda x, y: np.zeros((x.shape[0], y.shape[1]))
    
    # Preasignamos espacio    
    u     = np.zeros((Nx+1,Ny+1))    # Solución
    u_n   = np.zeros((Nx+1,Ny+1))    # Solución en t-dt
    u_nm1 = np.zeros((Nx+1,Ny+1))    # Solución en t-2*dt
    f_a   = np.zeros((Nx+1,Ny+1))    # Forzamiento
    sol   = np.zeros((Nx+1,Ny+1,Nt)) # Arreglo 3D para almacenar u
    
    It = list(range(0, t.shape[0]))
    
    # Condiciones iniciales
    u_n[:,:] = I(xv, yv)
    # primer paso de tiempo
    n = 0
    f_a[:,:] = f(xv, yv, t[n])  # Forzamiento
    V_a = V(xv, yv) 
    u = esquema(u, u_n, u_nm1, f_a,Cx2, Cy2, dt2, V=V_a, paso1=True)
    # reescribimos las matrices para el siguiente paso
    u_nm1, u_n, u = u_n, u, u_nm1
    # Resolvemos para los siguientes pasos de tiempo
    for n in It[1:-1]:
        f_a[:,:] = f(xv, yv, t[n])
        u = esquema(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2)
        u = np.where(mask, u, IBV)
        sol[:,:,n] = u
        u_nm1, u_n, u = u_n, u, u_nm1        
    return x, y, t, sol
           
def esquema(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2, V=None, paso1=False):
    if paso1:
        dt = np.sqrt(dt2)
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1
    u_xx = u_n[:-2,1:-1] - 2*u_n[1:-1,1:-1] + u_n[2:,1:-1]
    u_yy = u_n[1:-1,:-2] - 2*u_n[1:-1,1:-1] + u_n[1:-1,2:]
    u[1:-1,1:-1] = D1*u_n[1:-1,1:-1] - D2*u_nm1[1:-1,1:-1] + \
                   Cx2*u_xx + Cy2*u_yy + dt2*f_a[1:-1,1:-1]
    # if paso1:
    #     u[1:-1,1:-1] += dt*V[1:-1, 1:-1]
    # # Condiciones de frontera u=0

    u[:,0] = 0
    u[:,-1] = 0
    u[0,:] = 0
    u[-1,:] = 0
    
    return u

Lx = 20
Ly = 20
c = 1.0

Nx = 100
Ny=100
T = 40
dt = 0.1
V = None
f = None
IBV = 0
#C = 1

#I  = lambda x,y: np.sin(0.1*np.pi*x) * np.sin(0.1*np.pi*y)
I  = lambda x,y: np.exp(-(x-Lx/2)**2-(y-1)**2)*np.cos(np.pi*y/2)

def mask_circle(Nx,Ny,r):
    cx = Nx/2
    cy = Ny/2
    xm,ym = np.meshgrid(np.arange(Nx+1),np.arange(Ny+1))
    mask = np.sqrt((xm-cx)**2+(ym-cy)**2) >= r
    mask = mask.astype(bool)
    return mask

r=10
mask = mask_circle(Nx,Ny,r)


x, y, t, sol = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, mask, IBV)

m,n,t = sol.shape

X, Y = np.meshgrid(x,y)

#%%
def tiempos (dt,instante):
    N = int(instante/dt)
    return N
instantes= [5,10,12,20]
t_graf= np.zeros(len(instantes))

for j in range(len(instantes)):
    arg = 220+j+1
    t_graf[j]=int(tiempos(dt, instantes[j]))
    plt.figure(2)
    plt.subplot(arg)
    plt.pcolormesh(X,Y,sol[:,:,int(t_graf[j])],cmap = "viridis",shading = "auto",vmax=0.25)
    plt.suptitle("Onda en 2D",fontsize = 15)
    plt.text(0.65, 0.7, "tiempo ="+str(instantes[j]),color = "white",fontsize = 8)#+str(instantes[j-1]))
    plt.colorbar()


#%%
fig = plt.figure(1)
ax = fig.add_subplot(111)
plots=[]
for i in range(t):
    u = sol[:,:,i]
    u = np.where(mask, u, np.nan)
    pcolor= ax.pcolormesh(X, Y, u, shading = 'auto', cmap = 'viridis', vmax=0.2, vmin=-0.2)
    plots.append([pcolor])
ani = animation.ArtistAnimation(fig, plots, interval=20, blit=True)

plt.show()