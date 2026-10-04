#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:33:49 2022

@author: claudiovenegas

Perfiles de amortiguación CPML (condiciones de borde absorbentes, 2-D).
Define los arreglos de perfiles y los calcula con damp_profiles. Las funciones
que avanzan el campo (cpml_stress, cpml_vel, init_memory_*) están en CPMLD.py.
"""

import numpy as np
from math import exp, log, pi
from numba import config, njit, prange, set_num_threads
from CF import nx, ny, dt, dx, dy, vp

# MacBook Air M4: 10 hilos. Si la máquina tiene menos, se usan los disponibles.
NUM_HILOS = 10
set_num_threads(min(NUM_HILOS, config.NUMBA_NUM_THREADS))

damp = 1e-3
npow = 2.0
'''revisar topografia si coincide con los parametros del cpml'''
pml_points_x1 = 100
pml_points_x2 = 100
pml_points_y1 = 100
pml_points_y2 = 100

cp = vp


vel_damping = 1e-3
Rcoef = 0.001
f0 = 7.0
k_max_pml = 3.0
alpha_max_pml = 2.0*pi*(f0/2.0)
# d0 = -(N+1) * cp * ln(R) / (2 L)  (logaritmo NATURAL, como en Komatitsch & Martin)
d0x = -(npow + 1)*cp*log(Rcoef) / (2.*pml_points_x1*dx)
d0y = -(npow + 1)*cp*log(Rcoef) / (2.*pml_points_y1*dy)
a_x = np.zeros(nx*ny)
alpha_x = np.zeros(nx*ny)
k_x = np.ones(nx*ny)
b_x = np.zeros(nx*ny)
d_x = np.zeros(nx*ny)
a_y = np.zeros(nx*ny)
alpha_y = np.zeros(nx*ny)
k_y = np.ones(nx*ny)
b_y = np.zeros(nx*ny)
d_y = np.zeros(nx*ny)

a_x_half = np.zeros((nx-1)*(ny-1))
alpha_x_half = np.zeros((nx-1)*(ny-1))
k_x_half = np.ones((nx-1)*(ny-1))
b_x_half = np.zeros((nx-1)*(ny-1))
d_x_half = np.zeros((nx-1)*(ny-1))
a_y_half = np.zeros((nx-1)*(ny-1))
alpha_y_half = np.zeros((nx-1)*(ny-1))
k_y_half = np.ones((nx-1)*(ny-1))
b_y_half = np.zeros((nx-1)*(ny-1))
d_y_half = np.zeros((nx-1)*(ny-1))
vxout = np.zeros(nx*ny)
vyout = np.zeros(nx*ny)

sigmaxx_out = np.zeros((nx-1)*(ny-1))
sigmayy_out = np.zeros((nx-1)*(ny-1))
sigmaxy_out = np.zeros((nx-1)*(ny-1))
mem_dvx_dx = np.zeros((nx-1)*(ny-1))
mem_dvx_dy = np.zeros((nx-1)*(ny-1))
mem_dvy_dx = np.zeros((nx-1)*(ny-1))
mem_dvy_dy = np.zeros((nx-1)*(ny-1))
mem_dvx_dx_out = np.zeros((nx-1)*(ny-1))
mem_dvx_dy_out = np.zeros((nx-1)*(ny-1))
mem_dvy_dx_out = np.zeros((nx-1)*(ny-1))
mem_dvy_dy_out = np.zeros((nx-1)*(ny-1))


mem_dsigmaxx_dx = np.zeros((nx)*(ny))
mem_dsigmaxy_dx = np.zeros((nx)*(ny))
mem_dsigmaxy_dy = np.zeros((nx)*(ny))
mem_dsigmayy_dy = np.zeros((nx)*(ny))
mem_dsigmaxx_dx_out = np.zeros((nx)*(ny))
mem_dsigmaxy_dx_out = np.zeros((nx)*(ny))
mem_dsigmaxy_dy_out = np.zeros((nx)*(ny))
mem_dsigmayy_dy_out = np.zeros((nx)*(ny))


#inicializar en (fuera de la capa a=0 y d=0, por lo que estos valores no actúan):
alpha_x[:] = 1.1*alpha_max_pml
alpha_y[:] = 1.1*alpha_max_pml
b_x[:] = exp(-1.1*alpha_max_pml*dt)
b_y[:] = exp(-1.1*alpha_max_pml*dt)

alpha_x_half[:] = 1.1*alpha_max_pml
alpha_y_half[:] = 1.1*alpha_max_pml
b_x_half[:] = exp(-1.1*alpha_max_pml*dt)
b_y_half[:] = exp(-1.1*alpha_max_pml*dt)


@njit(fastmath=True, cache=True)
def _coef_cpml(pos, thick, d0, npow, k_max_pml, alpha_max_pml, dt):
    """Coeficientes CPML (d, k, alpha, b, a) a una distancia pos (en metros)
    del borde interior de la capa, de espesor thick."""
    x = pos/thick
    d = d0*x**npow
    k = 1. + (k_max_pml-1.)*x**npow
    '''Nota: el código komatish original escribe esto: T temp_alpha =
    alpha_max_pml * (1- (pos_pml / thick_pml_x1)) + 0.1 * d0 *
    alpha_max_pml, pero d0 no está definido en ninguna parte.
    Posteriormente, utiliza d0 = 1 con la cuadrícula
    de medios puntos. Entonces lo estamos usando como 1.'''
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
                  pml_points_y1, pml_points_y2, d0x, d0y, dx, dy, nx, ny, dt):
    '''Perfil de amortiguación. Aquí creamos diferentes vectores
    para la amortiguación.

    Posición dentro de la capa (metros, 0 en el borde interior y el espesor
    completo en el borde exterior de la malla):
        nodos   i=0..nx-1 en x = i*dx
        celdas  i=0..nx-2 en x = (i+1/2)*dx   (arreglos "half")
    d0x y d0y están calculados con el espesor de la capa 1 (izquierda/arriba);
    para la capa 2 se reescalan, ya que d0 es inversamente proporcional al
    espesor.'''

    thick_pml_x1 = pml_points_x1*dx
    thick_pml_x2 = pml_points_x2*dx
    thick_pml_y1 = pml_points_y1*dy
    thick_pml_y2 = pml_points_y2*dy
    d0x2 = d0x*pml_points_x1/pml_points_x2
    d0y2 = d0y*pml_points_y1/pml_points_y2

    #Perfiles en los nodos
    for j in prange(ny):
        for i in range(nx):
            if i < pml_points_x1:
                pos = (pml_points_x1-i)*dx
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_x1, d0x, npow, k_max_pml, alpha_max_pml, dt)
            elif i >= nx-pml_points_x2:
                pos = (i-(nx-1-pml_points_x2))*dx
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_x2, d0x2, npow, k_max_pml, alpha_max_pml, dt)
            else:
                temp_d = -1.
            if temp_d >= 0.:
                d_x[i+nx*j] = temp_d
                k_x[i+nx*j] = temp_k
                alpha_x[i+nx*j] = temp_alpha
                b_x[i+nx*j] = temp_b
                a_x[i+nx*j] = temp_a

            if j < pml_points_y1:
                pos = (pml_points_y1-j)*dy
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_y1, d0y, npow, k_max_pml, alpha_max_pml, dt)
            elif j >= ny-pml_points_y2:
                pos = (j-(ny-1-pml_points_y2))*dy
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_y2, d0y2, npow, k_max_pml, alpha_max_pml, dt)
            else:
                temp_d = -1.
            if temp_d >= 0.:
                d_y[i+nx*j] = temp_d
                k_y[i+nx*j] = temp_k
                alpha_y[i+nx*j] = temp_alpha
                b_y[i+nx*j] = temp_b
                a_y[i+nx*j] = temp_a

    #Perfiles en el centro de las celdas (para los esfuerzos)
    for j in prange(ny-1):
        for i in range(nx-1):
            if i < pml_points_x1:
                pos = (pml_points_x1-i-0.5)*dx
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_x1, d0x, npow, k_max_pml, alpha_max_pml, dt)
            elif i >= nx-1-pml_points_x2:
                pos = (i+0.5-(nx-1-pml_points_x2))*dx
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_x2, d0x2, npow, k_max_pml, alpha_max_pml, dt)
            else:
                temp_d = -1.
            if temp_d >= 0.:
                d_x_half[i+(nx-1)*j] = temp_d
                k_x_half[i+(nx-1)*j] = temp_k
                alpha_x_half[i+(nx-1)*j] = temp_alpha
                b_x_half[i+(nx-1)*j] = temp_b
                a_x_half[i+(nx-1)*j] = temp_a

            if j < pml_points_y1:
                pos = (pml_points_y1-j-0.5)*dy
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_y1, d0y, npow, k_max_pml, alpha_max_pml, dt)
            elif j >= ny-1-pml_points_y2:
                pos = (j+0.5-(ny-1-pml_points_y2))*dy
                temp_d, temp_k, temp_alpha, temp_b, temp_a = _coef_cpml(
                    pos, thick_pml_y2, d0y2, npow, k_max_pml, alpha_max_pml, dt)
            else:
                temp_d = -1.
            if temp_d >= 0.:
                d_y_half[i+(nx-1)*j] = temp_d
                k_y_half[i+(nx-1)*j] = temp_k
                alpha_y_half[i+(nx-1)*j] = temp_alpha
                b_y_half[i+(nx-1)*j] = temp_b
                a_y_half[i+(nx-1)*j] = temp_a

damp_profiles(a_x, a_x_half, a_y, a_y_half, d_x, k_x, alpha_x, b_x,
              d_x_half, k_x_half, alpha_x_half, b_x_half, d_y, k_y,
              alpha_y, b_y, d_y_half, k_y_half, alpha_y_half, b_y_half,
              npow, k_max_pml, alpha_max_pml, pml_points_x1, pml_points_x2,
              pml_points_y1, pml_points_y2, d0x, d0y, dx, dy, nx, ny, dt)
A = np.reshape(a_x, (ny, nx))
