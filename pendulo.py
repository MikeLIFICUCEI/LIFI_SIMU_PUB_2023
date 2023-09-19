# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:03:43 2023

@author: x
"""

###PENDULO SIMPLE
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

def pendulo(y,t,alpha,beta):
    theta,w= y
    dydt=[w, 
          -beta*w-alpha*np.sin(theta)]
    return dydt

#PARAMETROS CONSTANTES
g = 9.81
m = 1
L=2
b=[0,0.3,2*m*np.sqrt(g/L)]
#COEFICIENTES
beta_libre= b[0]/m
beta_sub = b[1]/m
beta_crit = b[2]/m
alpha=g/L
#TIEMPO
t_final = 30
t = np.linspace(0, t_final, t_final*10)
# CONDICIONES INICIALES
y0 = [np.pi/2,0]

#solucion
sol_libre = odeint(pendulo,y0,t,args=(alpha,beta_libre))
sol_sub = odeint(pendulo,y0,t,args=(alpha,beta_sub))
sol_crit = odeint(pendulo,y0,t,args=(alpha,beta_crit))

theta_libre = sol_libre[:,0] #toma la primer columna
w_libre = sol_libre[:,1]#toma la segunda columna

theta_sub = sol_sub[:,0] 
w_sub = sol_sub[:,1]

theta_crit = sol_crit[:,0] 
w_crit = sol_crit[:,1]

#graficas
fig, axs = plt.subplots(3)
fig.set_size_inches(8,6)
axs[0].plot(t,theta_libre,"tab:green")
axs[0].set_title("Libre")
axs[1].plot(t,theta_sub,"tab:orange")
axs[1].set_title("Amortiguamiento pequeño")
axs[2].plot(t,theta_crit,"tab:red")
axs[2].set_title("Amortiguamiento Crítico")

for ax in axs.flat:
    ax.set(xlabel='tiempo(s)', ylabel= r"$\theta $ (rad)")
for ax in axs.flat:
    ax.label_outer()
#ax.plot(t,w)




# Diagrama de fases
omega_f = np.linspace(-2*np.pi, 2*np.pi, 200)
theta_f = np.linspace(-2*np.pi, 2*np.pi, 200)

T,O = np.meshgrid(theta_f, omega_f)

Tp = O
Op_libre = -beta_libre*O - alpha*np.sin(T)
Op_sub = -beta_sub*O - alpha*np.sin(T)
Op_crit = -beta_crit*O - alpha*np.sin(T)


fig2=plt.figure(2)

plt.streamplot(T,O, Tp, Op_libre, color='gray')
plt.plot(theta_libre,w_libre,"tab:green")
plt.xlabel(r"$\theta $ (rad)", fontsize=13)
plt.ylabel(r"$\omega $ (rad/s)", fontsize=13)
plt.suptitle("Oscilación libre.",fontsize=16)



fig3=plt.figure(3)
plt.streamplot(T,O, Tp, Op_sub, color='gray')
plt.plot(theta_sub,w_sub,'tab:orange')
plt.xlabel(r"$\theta $ (rad)", fontsize=13)
plt.ylabel(r"$\omega $ (rad/s)", fontsize=13)
plt.suptitle("Amortiguamiento pequeño.",fontsize=16)


fig4=plt.figure(4)
plt.streamplot(T,O, Tp, Op_crit, color='gray')
plt.plot(theta_crit,w_crit,'tab:red')
plt.xlabel(r"$\theta $ (rad)", fontsize=13)
plt.ylabel(r"$\omega $ (rad/s)", fontsize=13)
plt.suptitle("Amortiguamiento crítico.",fontsize=16)


plt.show()