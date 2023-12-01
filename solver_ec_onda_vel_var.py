# -*- coding: utf-8 -*-

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
        c_max = max([c(x_) for x_ in np.linspace(0, L,101)])
        
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
        # calculamos c con la funcion
        c_0 = np.zeros(x.shape)
        for i in range(Nx+1):
            c_0[i] = c(x[i])
        c = c_0

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


L = 1 #longitud
C = 1 #numero de coulomb
L = 1

Nx = 100
V =  0 #velocidad inicial de la perturbacion
f = None
xc =0
sigma =0.05
beta = 1 #factor de seguridad
sf = 1.5 # factor de lentitud
medium = [0.4,0.7] # intervalo del medio
#Condiciones de frontera
U_0 = None
U_L = None
#condicion inicial
I = lambda x: np.exp(-((x-xc)/(2*sigma))**2)

c1 = 1
c2 = c1/sf
c3= c2/sf
dt = (L/Nx)/c1 #paso de tiempo
def c(x):
    cx = np.piecewise(x, [x < medium [0],\
                          ((x >= medium[0]) & (x < medium[1])), \
                              x >= medium[1]],\
                      [c1,c2,c3])
    return cx

Tao = 2*(medium[0]/c1 + (medium[1] - medium[0])/c2 + (L - medium[1])/c3)

u, x, t, sol = ec_onda(I, V, f, c, U_0, U_L, L, dt, C, Tao,beta)
#%%
m,n = sol.shape
x = np.linspace(0, L, n)
t =np.linspace(0, Tao, m)
X, T = np.meshgrid(x,t)

plt.figure(1)
plt.contourf(T,X,sol,20,cmap ='viridis')
plt.xlabel("Tiempo (t)")
plt.ylabel("Posición (m)")
plt.colorbar(label='Amplitud [m]')
#%%
def tiempos (dt,instante):
    N = int(instante/dt)
    return N
instantes= [0.01*Tao,0.20*Tao,0.5*Tao,0.99*Tao]
t_graf= np.zeros(len(instantes))

fig=plt.figure(figsize=(10,6))
for j in range(len(instantes)):
    arg = 220+j+1
    t_graf[j]=int(tiempos(dt, instantes[j]))
    plt.subplot(arg)
    plt.plot(x,sol[int(t_graf[j]),:],color = "blue")
    plt.plot(medium[0]*np.ones(20),np.linspace(-1,1,20),'red')
    plt.plot(medium[1]*np.ones(20),np.linspace(-1,1,20),'red')
    plt.ylim((-1,1))
    plt.title("Cuerda en tiempo = "+str(instantes[j])+"s")
    #plt.text(60, 0, "tiempo ="+str(instantes[j]),color = "k",fontsize = 12)#+str(instantes[j-1]))
    plt.xlabel("Posición en x (m)")
    plt.ylabel("Amplitud (m)")
    plt.grid()
    plt.tight_layout()   
#%%
fig =plt.figure(3,figsize=(8,5))
plts = []
for i in range(m):
    p, =plt.plot(sol[i,:],color="red")
    plts.append([p])
ani = animation.ArtistAnimation(fig, plts, interval = 50)

plt.show()