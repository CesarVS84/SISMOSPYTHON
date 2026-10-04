# -*- coding: utf-8 -*-
"""
CBA.py: perfiles de amortiguación de la CPML (condiciones de borde absorbentes).

crear_perfiles() devuelve todos los arreglos de perfiles (y los pone a cero/uno
fuera de la capa). No se ejecuta nada pesado al importar el módulo.

Convención (índice plano i + nx*j):
    - arreglos "en nodo"  : nx*ny            (para las velocidades)
    - arreglos "half"     : (nx-1)*(ny-1)    (centro de celda, para los esfuerzos)
Posición dentro de la capa (metros; 0 en el borde interior y el espesor completo
en el borde exterior de la malla):
    nodos  i = 0..nx-1  en x = i*dx
    celdas i = 0..nx-2  en x = (i+1/2)*dx
Un espesor de 0 desactiva esa capa.
"""

import numpy as np
from math import exp, log
from numba import njit, prange
import CF


@njit(fastmath=True, cache=True)
def _coef_cpml(pos, thick, d0, npow, k_max_pml, alpha_max_pml, dt):
    """Coeficientes CPML (d, k, alpha, b, a) a distancia pos (m) del borde
    interior de una capa de espesor thick (m)."""
    x = pos/thick
    d = d0*x**npow
    k = 1. + (k_max_pml-1.)*x**npow
    alpha = alpha_max_pml*(1.-x) + 0.1*alpha_max_pml
    b = exp(-(d/k + alpha)*dt)
    a = 0.
    if d > 1e-6:
        a = d*(b-1.)/(k*(d + k*alpha))
    return d, k, alpha, b, a


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def damp_profiles(a_x, a_x_half, a_y, a_y_half, d_x, k_x, alpha_x, b_x,
                  d_x_half, k_x_half, alpha_x_half, b_x_half, d_y, k_y,
                  alpha_y, b_y, d_y_half, k_y_half, alpha_y_half, b_y_half,
                  npow, k_max_pml, alpha_max_pml, pml_points_x1, pml_points_x2,
                  pml_points_y1, pml_points_y2, d0x1, d0x2, d0y1, d0y2,
                  dx, dy, nx, ny, dt):
    '''Perfiles de amortiguación. d0x1/d0x2/d0y1/d0y2 son el d0 de cada lado
    (izquierda, derecha, arriba, abajo); valen 0 si el espesor del lado es 0.'''

    thick_x1 = pml_points_x1*dx
    thick_x2 = pml_points_x2*dx
    thick_y1 = pml_points_y1*dy
    thick_y2 = pml_points_y2*dy

    #Perfiles en los nodos
    for j in prange(ny):
        for i in range(nx):
            if i < pml_points_x1:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((pml_points_x1-i)*dx, thick_x1, d0x1, npow, k_max_pml, alpha_max_pml, dt)
                d_x[i+nx*j] = t_d
                k_x[i+nx*j] = t_k
                alpha_x[i+nx*j] = t_al
                b_x[i+nx*j] = t_b
                a_x[i+nx*j] = t_a
            elif i >= nx-pml_points_x2:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((i-(nx-1-pml_points_x2))*dx, thick_x2, d0x2, npow, k_max_pml, alpha_max_pml, dt)
                d_x[i+nx*j] = t_d
                k_x[i+nx*j] = t_k
                alpha_x[i+nx*j] = t_al
                b_x[i+nx*j] = t_b
                a_x[i+nx*j] = t_a
            if j < pml_points_y1:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((pml_points_y1-j)*dy, thick_y1, d0y1, npow, k_max_pml, alpha_max_pml, dt)
                d_y[i+nx*j] = t_d
                k_y[i+nx*j] = t_k
                alpha_y[i+nx*j] = t_al
                b_y[i+nx*j] = t_b
                a_y[i+nx*j] = t_a
            elif j >= ny-pml_points_y2:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((j-(ny-1-pml_points_y2))*dy, thick_y2, d0y2, npow, k_max_pml, alpha_max_pml, dt)
                d_y[i+nx*j] = t_d
                k_y[i+nx*j] = t_k
                alpha_y[i+nx*j] = t_al
                b_y[i+nx*j] = t_b
                a_y[i+nx*j] = t_a

    #Perfiles en el centro de las celdas (para los esfuerzos)
    for j in prange(ny-1):
        for i in range(nx-1):
            q = i+(nx-1)*j
            if i < pml_points_x1:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((pml_points_x1-i-0.5)*dx, thick_x1, d0x1, npow, k_max_pml, alpha_max_pml, dt)
                d_x_half[q] = t_d
                k_x_half[q] = t_k
                alpha_x_half[q] = t_al
                b_x_half[q] = t_b
                a_x_half[q] = t_a
            elif i >= nx-1-pml_points_x2:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((i+0.5-(nx-1-pml_points_x2))*dx, thick_x2, d0x2, npow, k_max_pml, alpha_max_pml, dt)
                d_x_half[q] = t_d
                k_x_half[q] = t_k
                alpha_x_half[q] = t_al
                b_x_half[q] = t_b
                a_x_half[q] = t_a
            if j < pml_points_y1:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((pml_points_y1-j-0.5)*dy, thick_y1, d0y1, npow, k_max_pml, alpha_max_pml, dt)
                d_y_half[q] = t_d
                k_y_half[q] = t_k
                alpha_y_half[q] = t_al
                b_y_half[q] = t_b
                a_y_half[q] = t_a
            elif j >= ny-1-pml_points_y2:
                t_d, t_k, t_al, t_b, t_a = _coef_cpml((j+0.5-(ny-1-pml_points_y2))*dy, thick_y2, d0y2, npow, k_max_pml, alpha_max_pml, dt)
                d_y_half[q] = t_d
                k_y_half[q] = t_k
                alpha_y_half[q] = t_al
                b_y_half[q] = t_b
                a_y_half[q] = t_a


def d0_capa(cp, Rcoef, npow, espesor):
    """d0 = -(N+1) cp ln(R) / (2 L)  (logaritmo natural). 0 si L = 0."""
    if espesor <= 0:
        return 0.
    return -(npow + 1)*cp*log(Rcoef)/(2.*espesor)


def crear_perfiles(nx=CF.nx, ny=CF.ny, dx=CF.dx, dy=CF.dy, dt=CF.dt,
                   P1=CF.pml_points_x1, P2=CF.pml_points_x2,
                   Q1=CF.pml_points_y1, Q2=CF.pml_points_y2,
                   cp=CF.vp, npow=CF.npow, k_max_pml=CF.k_max_pml,
                   f0=CF.f0, Rcoef=CF.Rcoef):
    """Devuelve un diccionario con todos los perfiles de la CPML.

    cp es la velocidad P MÁXIMA del modelo. alpha_max = pi*f0 (f0 = frecuencia
    dominante de la fuente)."""
    assert P1 + P2 < nx and Q1 + Q2 < ny, 'La CPML no cabe en la malla'
    nn = nx*ny
    nc = (nx-1)*(ny-1)
    alpha_max_pml = np.pi*f0
    p = dict(
        a_x=np.zeros(nn), a_x_half=np.zeros(nc), a_y=np.zeros(nn), a_y_half=np.zeros(nc),
        d_x=np.zeros(nn), k_x=np.ones(nn), alpha_x=np.zeros(nn), b_x=np.zeros(nn),
        d_x_half=np.zeros(nc), k_x_half=np.ones(nc), alpha_x_half=np.zeros(nc), b_x_half=np.zeros(nc),
        d_y=np.zeros(nn), k_y=np.ones(nn), alpha_y=np.zeros(nn), b_y=np.zeros(nn),
        d_y_half=np.zeros(nc), k_y_half=np.ones(nc), alpha_y_half=np.zeros(nc), b_y_half=np.zeros(nc))
    #Fuera de la capa a = 0 y d = 0: estos valores no actúan, solo evitan ceros.
    for k in ('alpha_x', 'alpha_y', 'alpha_x_half', 'alpha_y_half'):
        p[k][:] = 1.1*alpha_max_pml
    for k in ('b_x', 'b_y', 'b_x_half', 'b_y_half'):
        p[k][:] = exp(-1.1*alpha_max_pml*dt)
    damp_profiles(p['a_x'], p['a_x_half'], p['a_y'], p['a_y_half'], p['d_x'], p['k_x'], p['alpha_x'], p['b_x'],
                  p['d_x_half'], p['k_x_half'], p['alpha_x_half'], p['b_x_half'], p['d_y'], p['k_y'],
                  p['alpha_y'], p['b_y'], p['d_y_half'], p['k_y_half'], p['alpha_y_half'], p['b_y_half'],
                  npow, k_max_pml, alpha_max_pml, P1, P2, Q1, Q2,
                  d0_capa(cp, Rcoef, npow, P1*dx), d0_capa(cp, Rcoef, npow, P2*dx),
                  d0_capa(cp, Rcoef, npow, Q1*dy), d0_capa(cp, Rcoef, npow, Q2*dy),
                  dx, dy, nx, ny, dt)
    return p
