#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:22:18 2022

@author: claudiovenegas

Modelo (topografía, material y fuente). Versión revisada.

Convenciones (las que ya usaba el código):
    - snif.npy es una matriz (NX, NY) = (501, 501) con TOP[fila, columna]:
      2 = aire, 1 = línea de la superficie (un nodo por columna), otro = roca.
      La fila crece hacia ARRIBA (el aire está en las filas mayores).
    - Los arreglos planos del solver se indexan i + nx*j con i = columna y
      j = fila.
    - too: bandera de dominio (nx+1)*(ny+1), indexada i + (nx+1)*j:
      0 debajo de la superficie, 1 sobre la superficie, 2 en el aire.
    - rho (nx*ny), lambdaa y mu ((nx-1)*(ny-1)): 0 en el aire.
"""

import numpy as np
from math import log10, sqrt


NX = NY = 501
dx = dy = 50
Lx = 25000
nx = NX-1
ny = NY-1


# Magnitud del sismo simulado: Mo = mu*A*U (N m); Mw = (2/3)(log10(Mo) - 9.1)
L = 2*dx
W = 1/3*L
A = L*W
mu0 = 3e10
U = 1
MO = mu0*A*U
Mw = (2/3)*(log10(MO)-9.1)
print('La magnitud del sismo simulado es de :', Mw, '(magnitud de momento Mw)')


TOP = np.load('snif.npy')
assert TOP.shape == (NX, NY), 'snif.npy debe ser de %dx%d' % (NX, NY)
d = TOP

# Fuente: par de nodos en diagonal (50 y 90), como en el original
F = np.zeros((nx, ny))
fa = [350, 293]
F[fa[0], fa[1]] = 50
F[fa[0]+1, fa[1]+1] = 90

sour1 = [fa[0], fa[1]]
sour2 = [fa[0]+1, fa[1]+1]

f = np.reshape(F, (nx*ny))

to = np.reshape(TOP, (NX*NY))

# Fila de la superficie en cada columna. Se exige exactamente un nodo con valor 1
# por columna (el código original lo suponía sin comprobarlo).
n_sup = (TOP == 1).sum(axis=0)
assert np.all(n_sup == 1), 'cada columna de snif.npy debe tener exactamente un nodo de superficie'
hc = np.argmax(TOP == 1, axis=0).astype(float)   # fila de la superficie (en nodos)
hb = hc*dx                                       # idem, en metros
hd = hb[::-1]/dx


# Bandera de dominio S[i, j] (i = columna, j = fila): 1 sobre la superficie, 2 sobre ella
jj = np.arange(ny+1)[None, :]
S = np.zeros((nx+1, ny+1))
S[jj == hc[:nx+1, None]] = 1
S[jj > hc[:nx+1, None]] = 2

# Nodo de la superficie bajo la fuente
G = [fa[0], int(hc[fa[0]])]


def dis(f, e):
    dd = np.sqrt((f[0]-e[0])**2+(f[1]-e[1])**2)*dx
    return dd
print('La profundidad de la falla es de ', dis(G, sour1)/1000, 'Km')


S = S.T
too = np.reshape(S, ((nx+1)*(ny+1)))


# Parámetros del material. Nota: cada celda (a, b) de lambda y mu se marca como roca
# según el nodo (a, b) de su esquina inferior, de modo que la fila de celdas que
# queda justo sobre la superficie también se considera roca. Es una imprecisión
# menor (aire con rigidez, sin masa) y se conserva para no cambiar el modelo.
Lam = np.where(TOP[:nx-1, :ny-1] != 2, 3e10, 0.)
Mu = np.where(TOP[:nx-1, :ny-1] != 2, 3e10, 0.)
Rho = np.where(TOP[:nx, :ny] != 2, 2800., 0.)     #rho compuacional

rho = np.reshape(Rho, (nx*ny))
mu = np.reshape(Mu, ((nx-1)*(ny-1)))
lambdaa = np.reshape(Lam, ((nx-1)*(ny-1)))
