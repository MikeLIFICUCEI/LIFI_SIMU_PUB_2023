# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:12:28 2023

@author: x
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

#sistema a resolver
def sistema(y,t,parametros):
    x1,v1,x2,v2=y
    m1,m2,k1,k2,b1,b2,L1,L2,mu1,mu2,F1,F2,w1,w2=parametros
    dydt=[v1,
          (-b1*v1-k1*(x1-L1)/m1+k2*(x2-x1-L2))/m1+mu1*(x1-L1)**3/m1+mu2*(x1-x2)**3/m1+F1*np.cos(t*w1),
          v2,
          (-b2*v2-k2*(x2-x1-L2))/m2+mu2*(x2-x1-L2)**3+F2*np.cos(t*w2)]
    return dydt

#constantes
m1 = 1
m2 = 1
k1 = 0.4
k2= 1.808
b1 = [0,1/10]
b2 = [0,1/5]
L1 = 0
L2 = 0
mu1= [-1/6,1/6]
mu2=[-1/10,1/10]
F1=[0,1/3]
F2=[0,1/5]
w1=1
w2=3/5

#parametros
parametros=[[m1,m2,k1,k2,b1[0],b2[0],L1,L2,mu1[0],mu2[0],F1[0],F2[0],w1,w2],
            [m1,m2,k1,k2,b1[1],b2[1],L1,L2,mu1[1],mu2[1],F1[1],F2[1],w1,w2]]
#condiciones iniciales
y0=[1,0,-0.5,0]
y0f=[0.7,0,0.1,0]
#tiempo
t1=np.linspace(0,50,1000)
t2=np.linspace(0,140,1000)
#Oscilaciones libres
sol_1=odeint(sistema,y0,t1, args = (parametros[0],))
x1=sol_1[:,0]
v1=sol_1[:,1]
x2=sol_1[:,2]
v2=sol_1[:,3]
#Oscilaciones forzadas
sol_f=odeint(sistema, y0f, t2, args=(parametros[1],))
x1f=sol_f[:,0]
v1f=sol_f[:,1]
x2f=sol_f[:,2]
v2f=sol_f[:,3]

#Potencia lista
def potencia(lista,p):
    y= [n**p for n in lista]
    return y

# Diagrama de fases
"""
omega_f = np.linspace(-2*np.pi, 2*np.pi, 200)
theta_f = np.linspace(-2*np.pi, 2*np.pi, 200)

V1,X1 = np.meshgrid(theta_f, omega_f)
V2,X2 =  np.meshgrid(theta_f, omega_f)
Xp1 = V1
Xp2 = V2
Vp_l1 = (-b1*V1-k1*(X1-L1)/m1+k2*(X2-X1-L2))/m1+mu1[0]*potencia((X1-L1),3)/m1+mu2[0]*potencia((X1-X2).3)/m1
Vp_l2 = (-b2*V2-k2*(X2-X1-L2))/m2+mu2[0]*potencia((X2-X1-L2),3)

Vp_f1 = (-b1*V1-k1*(X1-L1)/m1+k2*(X2-X1-L2))/m1+mu1[1]*potencia((X1-L1),3)/m1+mu2[1]*potencia((X1-X2),3)/m1+F1*np.cos(t1*w1)
Vp_f2 =  (-b2*V2-k2*(X2-X1-L2))/m2+mu2[1]*potencia((X2-X1-L2),3)+F2*np.cos(t2*w2)
"""
#GRAFICAS
fig1=plt.figure(1)
ax=fig1.add_subplot(212)
ax.plot(t1,x1,"tab:green")
ax.plot(t1,x2,"tab:red")
ax.set_xlabel("tiempo(s)", fontsize=13)
ax.set_ylabel("posición (m)", fontsize=13)


bx=fig1.add_subplot(221)

bx.plot(x1,v1,'tab:green')
#bx.set_xlabel(r"$\theta $ (rad)", fontsize=13)
bx.set_ylabel("velocidad (m/s)", fontsize=13)

cx=fig1.add_subplot(222)
cx.plot(x2,v2,'tab:red')
#cx.set_xlabel(r"$\theta $ (rad)", fontsize=13)
#cx.set_ylabel(r"$\omega $ (rad/s)", fontsize=13)

fig2=plt.figure(2)
ax=fig2.add_subplot(212)
ax.plot(t2,x1f,"tab:green")
ax.plot(t2,x2f,"tab:red")
ax.set_xlabel("tiempo(s)", fontsize=13)
ax.set_ylabel("posición (m)", fontsize=13)


bx=fig2.add_subplot(221)

bx.plot(x1f,v1f,'tab:green')
#bx.set_xlabel(r"$\theta $ (rad)", fontsize=13)
bx.set_ylabel("velocidad (m/s)", fontsize=13)

cx=fig2.add_subplot(222)
cx.plot(x2f,v2f,'tab:red')
#cx.set_xlabel(r"$\theta $ (rad)", fontsize=13)
#cx.set_ylabel(r"$\omega $ (rad/s)", fontsize=13)





