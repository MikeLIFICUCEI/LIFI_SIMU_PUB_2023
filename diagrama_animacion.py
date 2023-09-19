# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:51:00 2023

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

# Diagrama de fases
omega_f = np.linspace(-2*np.pi, 2*np.pi, 200)
theta_f = np.linspace(-2*np.pi, 2*np.pi, 200)

T,O = np.meshgrid(theta_f, omega_f)

Tp = O
Op = -beta*O - alpha*np.sin(T)

plt.figure(2)
plt.streamplot(T,O, Tp, Op, color='gray')
plt.plot(theta,omega, color='red')

########## Animación

x = L * np.sin(theta)
y = - L * np.cos(theta)

fig = plt.figure(10,figsize=(5,7))
ax = fig.add_subplot(2,1,1)
bx = fig.add_subplot(2,1,2)
bx.plot(x,t,'-b')
bx.set_ylabel('Tiempo [s]')
bx.set_xlabel('Pos. en X')
bx.set_ylim([t_tot,0])
bx.set_xlim([-2,2])

lns = []
for i in range(len(sol)):
    ln, = ax.plot([0, x[i]],[0,y[i]],color ='k', lw=2)
    po, = ax.plot(x[i],y[i],'ok',markersize = 15)
    tm  = ax.text(-1.8, 0.0, 'tiempo = %.1fs' % t[i])
    pu, = bx.plot([x[i]],[t[i]],'or',markersize = 10)
    lns.append([ln, po, tm, pu])
ax.set_xlim([-2,2])    
ax.set_ylim([-2,0.5])
ax.set_aspect('equal')    
ani = animation.ArtistAnimation(fig, lns, interval=50)
plt.tight_layout()