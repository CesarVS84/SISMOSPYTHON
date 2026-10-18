# -*- coding: utf-8 -*-
"""
TOP.py: modelo de densidades y rigideces, topografía y fuente.

La topografía viene de 'snif.npy' (matriz (501, 501), snif[fila, columna]: 2 = aire, 1 = superficie,
un nodo por columna; la fila crece hacia arriba) si el archivo existe, en la carpeta actual o en la
indicada por la variable de entorno SNIF_NPY; con él se usan también las estaciones y la fuente del
modelo del usuario (TOP.REAL = True). Si no existe, se usa un volcán sintético (cono truncado) con
estaciones propias, de modo que el código y las pruebas funcionan sin el archivo. Usa las mismas
convenciones que el TOP.py original:

    - j crece hacia ARRIBA: el aire está en los valores grandes de j.
    - superficie libre: en cada columna i, el nodo (i, hs[i]).
    - nodos con j >  hs[i] : aire        (bandera de dominio 2, velocidad nula)
    - nodos con j == hs[i] : superficie  (bandera 1)
    - nodos con j <  hs[i] : roca        (bandera 0)
    - arreglos planos con índice i + nx*j (i = columna, j = fila).

Arreglos que exporta (mismos nombres que usa el programa principal):
    too      ((nx+1)*(ny+1)) bandera de dominio, indexada i + (nx+1)*j
    rho      (nx*ny)          densidad en los nodos (0 en el aire)
    lambdaa  ((nx-1)*(ny-1))  lambda en las celdas (0 en el aire)
    mu       ((nx-1)*(ny-1))  módulo de corte en las celdas (0 en el aire)
    aa       (nx*ny)          1/(masa efectiva) de cada nodo (ver abajo)
    F        (nx, ny)         50 y 90: par de nodos de la fuente, en diagonal
    sour1    [i, j]           nodo de la estación "fuente"
    estaciones                lista [i, j] de las estaciones

Una celda es de roca si su fila superior de nodos está en la roca o sobre la
superficie. La masa de un nodo es proporcional a las celdas de roca que lo
rodean: aa = 4/(rho*n_celdas). Con aa = 1/rho en todos los nodos (como en el
programa original) la onda de Rayleigh viaja un 5 % más lenta.
"""

import os
import numpy as np
import CF
from CF import nx, ny, dx, dy, Rho, Lambda, Mu

TOP = None

# ------------------------------------------------------------------ Volcán
FILA_BASE = 330          # fila de la superficie lejos del volcán (llanura)
ALTURA = 50              # altura del volcán en nodos (2.5 km con dy = 50 m)
SEMIANCHO_BASE = 100     # nodos desde el centro hasta el pie del volcán
SEMIANCHO_CIMA = 10      # nodos de la meseta de la cima


def perfil_volcan(n=nx, fila_base=FILA_BASE, altura=ALTURA,
                  semiancho=SEMIANCHO_BASE, semiancho_cima=SEMIANCHO_CIMA):
    """Fila (real) de la superficie en cada columna: cono truncado."""
    i = np.arange(n)
    ic = n//2
    forma = np.clip((semiancho - np.abs(i-ic))/(semiancho - semiancho_cima), 0., 1.)
    return fila_base + altura*forma


def modelo_desde_superficie(hs, nx=nx, ny=ny, rho0=Rho, lam0=Lambda, mu0=Mu):
    """Construye el modelo a partir de la fila (real) de la superficie hs[i].

    Devuelve un diccionario con js (fila entera de la superficie), too, rho,
    lambdaa, mu, aa."""
    js = np.rint(hs).astype(np.int64)
    js_ext = np.concatenate([js, js[-1:]])           # too tiene nx+1 columnas
    jn = np.arange(ny+1)[:, None]
    too = np.zeros((ny+1, nx+1))
    too[jn > js_ext[None, :]] = 2.
    too[jn == js_ext[None, :]] = 1.
    # celdas: roca si su fila superior de nodos no sobrepasa la línea de la superficie
    zc = 0.5*(js[:-1] + js[1:])
    jc = np.arange(ny-1)[:, None]
    roca = (jc + 1 <= zc[None, :])                   # [j, i]
    lambdaa = np.where(roca, lam0, 0.).ravel()
    mu = np.where(roca, mu0, 0.).ravel()
    cuenta = np.zeros((ny, nx))                      # celdas de roca que tocan cada nodo
    cuenta[:-1, :-1] += roca
    cuenta[:-1, 1:] += roca
    cuenta[1:, :-1] += roca
    cuenta[1:, 1:] += roca
    rho = np.where(cuenta > 0, rho0, 0.)
    aa = np.zeros((ny, nx))
    aa[cuenta > 0] = 4./(rho0*cuenta[cuenta > 0])
    return dict(js=js, too=too.ravel(), rho=rho.ravel(), lambdaa=lambdaa, mu=mu,
                aa=aa.ravel(), cuenta=cuenta.ravel())


# ------------------------------------------------------------------- Modelo
def _ruta_snif():
    for r in (os.environ.get('SNIF_NPY'), 'snif.npy', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snif.npy')):
        if r and os.path.isfile(r):
            return r
    return None


RUTA_SNIF = _ruta_snif()
REAL = RUTA_SNIF is not None
if REAL:
    _snif = np.load(RUTA_SNIF, allow_pickle=False)
    assert _snif.shape == (nx+1, ny+1), 'snif.npy debe ser de %dx%d' % (nx+1, ny+1)
    assert np.all((_snif == 1).sum(axis=0) == 1), 'cada columna de snif.npy debe tener un nodo de superficie'
    hs = np.argmax(_snif == 1, axis=0)[:nx].astype(float)             # fila de la superficie por columna
else:
    hs = perfil_volcan()
_m = modelo_desde_superficie(hs)
js = _m['js']
too = _m['too']
rho = _m['rho']
lambdaa = _m['lambdaa']
mu = _m['mu']
aa = _m['aa']

# ------------------------------------------------------- Fuente (par diagonal)
# Como en el TOP.py original: F = 50 en el nodo fa y F = 90 en (fa+1, fa+1)
# (velocidades opuestas, ver CPML.py), a 3 km bajo la cima (60 nodos).
ic = nx//2
fa = [350, int(js[350]) - 60] if REAL else [ic, int(js[ic]) - 60]    # modelo real: [350, 293]
F = np.zeros((nx, ny))
F[fa[0], fa[1]] = 50
F[fa[0]+1, fa[1]+1] = 90
f = np.reshape(F, (nx*ny))
sour1 = [fa[0], fa[1]]
sour2 = [fa[0]+1, fa[1]+1]


# ------------------------------------------------------------- Estaciones
def _sup(i):
    return [i, int(js[i])]


if REAL:       # estaciones del usuario (E1-E7 sobre la superficie, E8 y E9 en profundidad)
    estaciones = [[170, 356], [190, 360], [210, 368], [261, 392], [281, 381], [301, 369],
                  [321, 361], [200, 250], [300, 250]]
    assert all(e[1] == js[e[0]] for e in estaciones[:7]), 'E1-E7 deben estar sobre la superficie de snif.npy'
else:
    estaciones = [
        _sup(ic-90), _sup(ic-60), _sup(ic-30),     # flanco izquierdo (sube hacia la derecha)
        _sup(ic+30), _sup(ic+60), _sup(ic+90),     # flanco derecho (baja hacia la derecha)
        _sup(ic+130),                              # llanura
        [ic-50, int(js[ic]) - 80],                 # dos estaciones en profundidad
        [ic+50, int(js[ic]) - 80],
    ]
