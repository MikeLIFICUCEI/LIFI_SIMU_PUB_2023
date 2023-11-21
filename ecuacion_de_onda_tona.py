#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:12:24 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%
def ec_onda(I , V , f, c ,U_0 , U_L , L , dt , T , C = 1):
    '''
    I:      Condiciones iniciales.
    V:      Velocidad inicial de la perturbación.
    f:      forzamiento .
    c:      Velocidad de propagación.
    U_0:    Condiciones de frontera.
    U_L:    Condiciones de frontera.
    L:      Longitud.
    dt:     Diferencial de tiempo 
    C:      Número de coulomb 
    T:      Tiempo total de la simulación
    '''
    
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
    if I is None or I == 0:
        I = (lambda x: 0)    
    # implementación de condiciones de frontera como función.    
    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0    
    # Preasignamos espacio para la solución    
    u = np.zeros(Nx+1) # Solución
    u_n = np.zeros(Nx+1) # Solución a 1 nivel de tiempo atrás
    u_nm1 = np.zeros(Nx+1) # Solución a 2 niveles de tiempo atrás
    
    # creamos índices para la notación de índices
    Ix = list(range(0, Nx+1))
    It = list(range(0, Nt+1))

    # Insertamos condiciones iniciales en u_n (se puede hacer con copy).
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    # Para el primer paso de tiempo
    for i in Ix[1:-1]:
        u[i] = u_n[i] - 0.5*C2*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
               0.5*dt**2*f(x[i], t[0])  + dt*V(x[i])

    i = Ix[0]
    # forntera izquierda
    if U_0 is None:
        # Se imponen las condiciones de frontera du/dn = 0
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        u[i] = u_n[i] - 0.5*C2*(u_n[im1] - 2*u_n[i] + u_n[ip1]) + \
               0.5*dt**2*f(x[i], t[0]) + dt*V(x[i])
    else:
        u[0] = U_0(dt)
    #frontera derecha    
    i = Ix[-1]
    if U_L is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        u[i] = u_n[i] - 0.5*C2*(u_n[im1] - 2*u_n[i] + u_n[ip1]) + \
               0.5*dt**2*f(x[i], t[0]) + dt*V(x[i])
    else:
        u[i] = U_L(dt)
    
    # Cambiamos las variables antes del siguiente paso de tiempo.    
    u_nm1, u_n, u = u_n, u, u_nm1
    
    sol = np.zeros((Nt+1,Nx+1)) # preasignamos espacio para almacenar la sol.
    
    for n in It[1:-1]:
        for i in Ix[1:-1]:
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   C2*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
                   dt**2*f(x[i], t[n])
                      
        # Insertamos condiciones de frontera
        i = Ix[0]
        if U_0 is None:
            # Imponemos valores en la frontera
            ip1 = i+1
            im1 = ip1
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   C2*(u_n[im1] - 2*u_n[i] + u_n[ip1]) + \
                   dt**2*f(x[i], t[n])
        else:
            u[0] = U_0(t[n+1])

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   C2*(u_n[im1] - 2*u_n[i] + u_n[ip1]) + \
                   dt**2*f(x[i], t[n])
        else:
            u[i] = U_L(t[n+1])
        # Cambiamos variables antes del siguiente paso de tiempo.
        u_nm1, u_n, u = u_n, u, u_nm1
        sol[n] = u  
    return u, x, t, sol
#%%
A=1
V=0
c=1
L=10
dt=0.1
m=2
T=40

#%% Actividad 1

I=lambda x:A*np.sin(np.pi/L*m*x)
f=None

U_0=0
U_L=0

#%% Actividad 2
I =lambda x: np.exp(-(x-5)**2/(0.25))
f =lambda x,t: 0.05*np.sin(np.pi/L*2*x)

U_0=None
U_L=None

#%%

u,x,t,sol=ec_onda(I, V, f, c, U_0, U_L, L, dt, T)
m,n=sol.shape
x=np.linspace(0,L,n)
t=np.linspace(0,T,m)
X,T_s=np.meshgrid(x,t)
#%%
#plt.close()

plt.figure(1)
plt.contour(T_s,X,sol,1000,cmap='viridis')
plt.ylabel('Distnacia [m]')
plt.xlabel('Tiempo [s]')
plt.colorbar(label='Amplitud [m]')
plt.show()
plt.savefig('actividad_2_espacio-tiempo.png')  
#%%
#plt.close()

fig=plt.figure(2)
plts=[]
for i in range(m):
    p,=plt.plot(sol[i,:],color='k')
    plts.append([p])
ani= animation.ArtistAnimation(fig, plts,interval=50)
plt.ylabel('Amplitud')
#%%
#plt.close()

fig=plt.figure(3,figsize=(8,5))
times=np.array([(T/dt)*0.05,(T/dt)*0.25,(T/dt)*0.50,(T/dt)*0.85])

for i in range(len(times)):
    plt.subplot(2,2,i+1)
    plt.plot(sol[int(times[i]),:],color='k')
    plt.title("Cuerda en el instante: "+str(times[i]*dt)+'s')
    plt.xlabel('Distancia [cm]')
    plt.ylabel('Amplitud [m]')
    plt.ylim((-0.7955128988443754,1.4528772682467586))
    plt.grid(True)
    
plt.tight_layout()
plt.show()
plt.savefig('actividad_2_animación.png')  
#%%
