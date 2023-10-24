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

def ec_onda(I, V, f, c, L, dt, C, T):
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1) # Puntos de malla en tiempo
    dx = dt*c/C
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1) # Puntos de malla en espacio
    C2 = C**2 # C cuadrada nada más
    # Para f = 0 y V = 0
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
    # Preasignamos espacio para la solución    
    u = np.zeros(Nx+1) # Solución
    u_n = np.zeros(Nx+1) # Solución a 1 nivel de tiempo atrás
    u_nm1 = np.zeros(Nx+1) # Solución a 2 niveles de tiempo atrás

    # Insertamos condiciones iniciales en u_n.
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    # Para el primer paso de tiempo
    n = 0 #para f(x,0)
    for i in range(1, Nx):
        u[i] = u_n[i] + dt*V(x[i]) + 0.5*C2*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
               0.5*dt**2*f(x[i], t[n])
    u[0] = u[Nx]; u[Nx] = u[0]
    # Cambiamos las variables antes del siguiente paso de tiempo.
    u_nm1[:] = u_n; u_n[:] = u
    sol = np.zeros((Nt,len(u))) # preasignamos espacio para almacenar la sol.
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2*u_n[i] + C2*(u_n[i+1] - 2*u_n[i] + u_n[i-1]) + \
                   dt**2*f(x[i], t[n])
        # Insertamos condiciones de frontera
        u[0] = u[Nx]; u[Nx] = u[0]
        # Cambiamos variables antes del siguiente paso de tiempo.
        u_nm1[:] = u_n; u_n[:] = u
        sol[n] = u
    return u, x, t, sol

#PARAMETROS
c = 1#velocidad de propagacion
L = 10 #longitud
dt = 0.1 
C = 1 #numero de coulomb
T = 40
V =  0 #velocidad inicial de la perturbacion
#f = 0 #forzamiento
f = lambda x,t: 0.4*np.sin((np.pi/L)*3*x)

#condicion inicial
I = lambda x: np.exp(-((x-5)**2)/0.25)

u, x, t, sol = ec_onda(I, V, f, c, L, dt, C, T)
m,n = sol.shape
x = np.linspace(0, L, n)
t =np.linspace(0, T, m)
X, T = np.meshgrid(x,t)

plt.figure(1)
plt.contourf(T,X,sol,50,cmap ='inferno')

fig =plt.figure()
plts = []
for i in range(m):
    p, =plt.plot(sol[i,:],color = 'red')
    plts.append([p])
ani = animation.ArtistAnimation(fig, plts, interval = 50)

plt.show()



