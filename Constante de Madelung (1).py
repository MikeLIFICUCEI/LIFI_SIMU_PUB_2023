h# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:42:17 2023

@author: pepet
"""
#cáculo de la constante de Madelung
import numpy as np

# Inicializar variables
n = 100000000
v = np.array([0, 0, 0])
H = np.zeros((n, 3))
vn = np.zeros(n)
signo = np.zeros(n)
T = np.zeros(n)

# Ciclo para obtener las configuraciones de posición
for i in range(n):
    if v[0] == v[1] and v[2] == v[1]:
        v[0] += 1
        v[1] = 0
        v[2] = 0
    elif v[1] == v[2] and v[0] != v[1]:
        v[0] = v[0]
        v[1] += 1
        v[2] = 0
    elif v[1] != v[2]:
        v[2] += 1
    H[i, :] = v

# Calcular las normas de H en un solo paso
nom = np.linalg.norm(H, axis=1)

# Ciclo para obtener las combinaciones posibles de cada configuración
for i in range(len(H)):
    if H[i, 1] == 0 and H[i, 2] == 0:
        N = 6
    elif H[i, 1] == H[i, 0] and H[i, 2] == H[i, 0] and H[i, 2] == H[i, 2]:
        N = 8
    elif H[i, 1] == H[i, 0] and H[i, 2] == 0 and H[i, 2] != H[i, 0]:
        N = 12
    elif H[i, 1] == H[i, 0] and H[i, 2] > 0 and H[i, 2] != H[i, 0]:
        N = 24
    elif H[i, 1] == H[i, 2] and H[i, 2] > 0 and H[i, 2] != H[i, 0]:
        N = 24
    elif H[i, 0] != H[i, 1] and H[i, 2] == 0 and H[i, 1] > 0:
        N = 24
    elif H[i, 0] == H[i, 2] and H[i, 2] != H[i, 2] != H[i, 1] and H[i, 1] > 0:
        N = 24
    elif H[i, 0] != H[i, 1] and H[i, 1] != H[i, 2] and H[i, 0] != H[i, 2] and H[i, 0] > 0 and H[i, 1] > 0 and H[i, 2] > 0:
        N = 48
    else:
        N = 0
    vn[i] = N

# Calcular el signo para todos los elementos en un solo paso
signo = (-1) ** (H[:, 0] + H[:, 1] + H[:, 2])

# Calcular la suma en un solo paso
T = (vn / nom) * signo
SUMA = np.sum(T)
print(f'SUMA: {SUMA}')
