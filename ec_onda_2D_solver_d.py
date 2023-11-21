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

def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T):
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
    if I is None or I == 0:
        I = lambda x, y: np.zeros((x.shape[0], y.shape[1]))    
    
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
    #V_a = V(xv, yv) 
    u = esquema(u, u_n, u_nm1, f_a,Cx2, Cy2, dt2, V=None, paso1=True)
    # reescribimos las matrices para el siguiente paso
    u_nm1, u_n, u = u_n, u, u_nm1
    # Resolvemos para los siguientes pasos de tiempo
    for n in It[1:-1]:
        f_a[:,:] = f(xv, yv, t[n])
        u = esquema(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2)
        sol[:,:,n] = u
        u_nm1, u_n, u = u_n, u, u_nm1        
    return x, y, t, sol
           
def esquema(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2, V=None, paso1=False):
    if paso1:
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2  
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1
    u_xx = u_n[:-2,1:-1] - 2*u_n[1:-1,1:-1] + u_n[2:,1:-1]
    u_yy = u_n[1:-1,:-2] - 2*u_n[1:-1,1:-1] + u_n[1:-1,2:]
    u[1:-1,1:-1] = D1*u_n[1:-1,1:-1] - D2*u_nm1[1:-1,1:-1] + \
                   Cx2*u_xx + Cy2*u_yy + dt2*f_a[1:-1,1:-1]

    # Condiciones de frontera u=0
    j = 0 # primer vector
    u[:,j] = 0
    j = -1 # último 
    u[:,j] = 0
    
    i = 0
    u[i,:] = 0
    i = -1
    u[i,:] = 0
 
    return u
Lx = 10
Ly = 10
c = 1.0

Nx = 80
Ny=80
T = 40
dt = 0.05 

#I  = lambda x,y: np.sin(0.1*np.pi*x) * np.sin(0.1*np.pi*y)
I  = lambda x,y: np.sin(0.1*np.pi*x) * np.sin(0.1*np.pi*y)

x, y, t, sol = solver(I, None, None, c, Lx, Ly, Nx, Ny, dt, T)

m,n,t = sol.shape
X,Y = np.meshgrid(x,y)

fig = plt.figure(1)
ax = fig.add_subplot(111)

plts = []
for i in range(t):
    u = sol[:,:,i]
    pcolor= ax.pcolormesh(X,Y,u,shading = 'auto',
                          cmap = 'seismic', vmin = -0.3, vmax = 0.3)
    plts.append([pcolor])
ani = animation.ArtistAnimation(fig, plts, interval = 10, blit =True)
plt.show
 