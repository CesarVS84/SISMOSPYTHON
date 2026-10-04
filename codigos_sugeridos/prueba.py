# -*- coding: utf-8 -*-
"""
prueba.py: pendientes de la superficie y tipo de transición (banderas) para las
condiciones de borde de Jih (CBO.py).

Se usa la misma clasificación del prueba.py original, con tan = pendiente del
tramo (positiva si la superficie SUBE hacia la derecha):

    1: pendiente constante, tan >= 2        2: pendiente constante, tan < 2
    3: cóncavo, horizontal -> pendiente (tan_der >= 1)
    4: cóncavo, pendiente suave -> empinada
    5: cambio convexo de pendiente          6: convexo, pendiente -> horizontal

Las fórmulas de CBO.py están escritas para tramos que SUBEN hacia la derecha. Los
que bajan se tratan reflejando el perfil en x (segunda pasada, si = -1). Las cimas
y los valles (cambio de signo de la pendiente) no tienen caso: bandera 0, y esos
nodos quedan solo con el aire de densidad y rigidez nulas de TOP.py.

Arreglos que exporta (nx*ny):
    elevation, boundary_flag_velocity            primera pasada (si = +1)
    elevation_izq, boundary_flag_velocity_izq    segunda pasada (si = -1)
elevation[i + nx*j] es el ángulo del tramo que va de la columna i a la i+1, y vale
lo mismo en toda la columna. En la segunda pasada es el del tramo i-1 -> i.
"""

import numpy as np
from CF import nx, ny, dx, dy
import TOP


def clasificar(js):
    """Tipo de transición y ángulo en cada columna de un perfil js (fila de la
    superficie por columna). Devuelve (tipo[nx], angulo[nx]); angulo[c] es el del
    tramo c -> c+1."""
    n = len(js)
    tan_tramo = np.diff(js)*dy/dx                 # tramo c -> c+1
    angulo = np.zeros(n)
    angulo[:-1] = np.arctan(tan_tramo)
    tipo = np.zeros(n)
    tl = np.zeros(n)
    tr = np.zeros(n)
    tl[1:-1] = tan_tramo[:-1]
    tr[1:-1] = tan_tramo[1:]
    d = tr - tl
    ok = np.zeros(n, dtype=bool)
    ok[1:-1] = True
    ok &= (tl >= 0) & (tr >= 0)                   # solo tramos que suben hacia la derecha
    tipo[ok & (d == 0) & (tl >= 2.)] = 1
    tipo[ok & (d == 0) & (tl < 2.)] = 2
    tipo[ok & (d > 0) & (tl == 0) & (tr >= 1.)] = 3
    tipo[ok & (d > 0) & (tl < tr) & (tl != 0)] = 4
    tipo[ok & (d < 0) & (tr > 0)] = 5
    tipo[ok & (d < 0) & (tr == 0)] = 6
    return tipo, angulo


def _arreglos(tipo, angulo, js):
    """Pasa tipo/ángulo por columna a arreglos nx*ny."""
    el = np.zeros((ny, nx))
    el[:, :] = angulo[None, :]
    fl = np.zeros((ny, nx))
    cols = np.arange(nx)
    fl[js[cols], cols] = tipo
    return el.ravel(), fl.ravel()


identificador_de_pendientes = None

# Primera pasada: tramos que suben hacia la derecha
_t1, _a1 = clasificar(TOP.js.astype(float))
elevation, boundary_flag_velocity = _arreglos(_t1, _a1, TOP.js)
identificador_de_pendientes = _t1

# Segunda pasada: perfil reflejado en x (los tramos que bajan pasan a subir)
_t2, _a2 = clasificar(TOP.js[::-1].astype(float))
_t2 = _t2[::-1]
_a2r = np.zeros(nx)
_a2r[1:] = _a2[::-1][:-1]                       # angulo2[c] = ángulo del tramo (c-1 -> c) reflejado
_t2 = np.where(_t1 > 0, 0., _t2)                # los nodos planos ya los cubre la primera pasada
elevation_izq, boundary_flag_velocity_izq = _arreglos(_t2, _a2r, TOP.js)

elevation_values = elevation
