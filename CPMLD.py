#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 20:28:14 2021

@author: claudiovenegas

CPML (Convolutional PML) para elastodinámica 2-D. Versión revisada.

Convenciones de memoria (todas con índice plano i + nx*j):
    - velocidades y perfiles "en nodo"      : nx*ny
    - esfuerzos interiores y perfiles "half": (nx-1)*(ny-1)  (celda i entre los
                                              nodos i e i+1)
    - esfuerzos con borde (Sbxx, Sbxy, Sbyy): (nx+1)*(ny+1)  (Sb[i] = celda i-1)

Los coeficientes que reciben cpml_stress (b, c, d) y cpml_vel (aa) deben ser
(lambda+2mu), lambda, mu y 1/rho SIN dividir por dx: estas funciones ya dividen
por dx/dy al calcular las derivadas. (El programa principal usa para el
interior los mismos arreglos pero ya divididos por dx, por lo que debe pasar
b*dx, c*dx, d*dx y aa*dx a estas funciones.)

Cada función recorre la malla una sola vez (un hilo por fila de la malla),
sin tratar bordes y esquinas por separado.
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


vel_damping = 5e-3
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


'''CPML en sí. Esta función cambia velocidades y tensiones en 2-D. Su estructura
es la siguiente.
CPML vacío (tensiones, velocidades, tensiones de memoria, velocidades de memoria
parámetros de absorción de capa, parametros fisicos(densidad, coeficientes de Lame)
parametros de la malla).
Esta función debe funcionar en 4 capas diferentes: superior, inferior, izquierda
y derecha. En todos los casos, la función cambia alguna cantidad solo dentro de
la capa.

Una celda (o nodo) puede pertenecer a la capa en x, en y, o en ambas (esquina).
En cada caso se actualiza la memoria de las derivadas que corresponda y solo
una vez por paso de tiempo.'''


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def cpml_stress(sigmaxx_out, sigmaxy_out, sigmayy_out, sigmaxx, sigmaxy, sigmayy,
                vx, vy, mem_dvx_dx, mem_dvy_dx, mem_dvx_dy, mem_dvy_dy,
                mem_dvx_dx_out, mem_dvy_dx_out, mem_dvx_dy_out, mem_dvy_dy_out,
                a_x_half, a_y_half, k_x_half, alpha_x_half, b_x_half, k_y_half,
                alpha_y_half, b_y_half, pml_points_x1, pml_points_x2,
                pml_points_y1, pml_points_y2, b, c, d, dx, dy, dt, nx, ny):
    '''Actualiza los esfuerzos de las celdas dentro de la capa.
    sigma*_out debe ser un arreglo distinto de sigma* y vx, vy son las
    velocidades nuevas.'''

    for j in prange(ny-1):
        en_y = j < pml_points_y1 or j >= ny-1-pml_points_y2
        for i in range(nx-1):
            en_x = i < pml_points_x1 or i >= nx-1-pml_points_x2
            if not (en_x or en_y):
                continue
            q = i+(nx-1)*j

            #Define derivadas para las velocidades:
            dvxdx = (0.5/dx)*(vx[i+1+nx*j]-vx[i+nx*j]+vx[i+1+nx*(j+1)]-vx[i+nx*(j+1)])
            dvydx = (0.5/dx)*(vy[i+1+nx*j]-vy[i+nx*j]+vy[i+1+nx*(j+1)]-vy[i+nx*(j+1)])
            dvxdy = (0.5/dy)*(vx[i+nx*(j+1)]-vx[i+nx*j]+vx[i+1+nx*(j+1)]-vx[i+1+nx*j])
            dvydy = (0.5/dy)*(vy[i+nx*(j+1)]-vy[i+nx*j]+vy[i+1+nx*(j+1)]-vy[i+1+nx*j])

            if en_x:
                #Cambiando la memoria de las derivadas
                mem_dvx_dx_out[q] = b_x_half[q]*mem_dvx_dx[q] + a_x_half[q]*dvxdx
                mem_dvy_dx_out[q] = b_x_half[q]*mem_dvy_dx[q] + a_x_half[q]*dvydx
                #Cambiando las derivadas
                dvxdx = (1./k_x_half[q])*dvxdx + mem_dvx_dx_out[q]
                dvydx = (1./k_x_half[q])*dvydx + mem_dvy_dx_out[q]

            if en_y:
                mem_dvx_dy_out[q] = b_y_half[q]*mem_dvx_dy[q] + a_y_half[q]*dvxdy
                mem_dvy_dy_out[q] = b_y_half[q]*mem_dvy_dy[q] + a_y_half[q]*dvydy
                dvxdy = (1./k_y_half[q])*dvxdy + mem_dvx_dy_out[q]
                dvydy = (1./k_y_half[q])*dvydy + mem_dvy_dy_out[q]

            #Cambiando los stresses:
            sigmaxx_out[q] = sigmaxx[q] + dt*(b[q]*dvxdx + c[q]*dvydy)
            sigmayy_out[q] = sigmayy[q] + dt*(b[q]*dvydy + c[q]*dvxdx)
            sigmaxy_out[q] = sigmaxy[q] + dt*d[q]*(dvxdy + dvydx)


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def cpml_vel(vxout, vyout, sigmaxx, sigmaxy, sigmayy, vx, vy, mem_dsigmaxx_dx,
             mem_dsigmaxy_dx, mem_dsigmaxy_dy, mem_dsigmayy_dy, mem_dsigmaxx_dx_out,
             mem_dsigmaxy_dx_out, mem_dsigmaxy_dy_out, mem_dsigmayy_dy_out,
             a_x, a_y, k_x, alpha_x, b_x, k_y, alpha_y, b_y, pml_points_x1,
             pml_points_x2, pml_points_y1, pml_points_y2, aa, dx, dy, dt,
             vel_damping, nx, ny):
    '''Actualiza las velocidades de los nodos dentro de la capa. sigma* son los
    esfuerzos con borde (nx+1)*(ny+1); vx, vy son las velocidades del paso
    anterior (arreglos distintos de vxout, vyout) y aa = 1/rho por nodo.
    Al final impone velocidad nula en el borde de la malla (Dirichlet).'''

    for j in prange(ny):
        en_y = j < pml_points_y1 or j >= ny-pml_points_y2
        for i in range(nx):
            p = i+nx*j
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                #Condiciones de borde para las velocidades (Dirichlet)
                vxout[p] = 0.
                vyout[p] = 0.
                continue

            en_x = i < pml_points_x1 or i >= nx-pml_points_x2
            if not (en_x or en_y):
                continue

            #Definiendo las derivadas
            dsigmaxxdx = (0.5/dx)*(sigmaxx[i+1+(nx+1)*j]-sigmaxx[i+(nx+1)*j]+sigmaxx[i+1+(nx+1)*(j+1)]-sigmaxx[i+(nx+1)*(j+1)])
            dsigmaxydx = (0.5/dx)*(sigmaxy[i+1+(nx+1)*j]-sigmaxy[i+(nx+1)*j]+sigmaxy[i+1+(nx+1)*(j+1)]-sigmaxy[i+(nx+1)*(j+1)])
            dsigmaxydy = (0.5/dy)*(sigmaxy[i+(nx+1)*(j+1)]-sigmaxy[i+(nx+1)*j]+sigmaxy[i+1+(nx+1)*(j+1)]-sigmaxy[i+1+(nx+1)*j])
            dsigmayydy = (0.5/dy)*(sigmayy[i+(nx+1)*(j+1)]-sigmayy[i+(nx+1)*j]+sigmayy[i+1+(nx+1)*(j+1)]-sigmayy[i+1+(nx+1)*j])

            if en_x:
                #Cambiando las memorias
                mem_dsigmaxx_dx_out[p] = b_x[p]*mem_dsigmaxx_dx[p] + a_x[p]*dsigmaxxdx
                mem_dsigmaxy_dx_out[p] = b_x[p]*mem_dsigmaxy_dx[p] + a_x[p]*dsigmaxydx
                #Cambiando las derivadas
                dsigmaxxdx = (1./k_x[p])*dsigmaxxdx + mem_dsigmaxx_dx_out[p]
                dsigmaxydx = (1./k_x[p])*dsigmaxydx + mem_dsigmaxy_dx_out[p]

            if en_y:
                mem_dsigmaxy_dy_out[p] = b_y[p]*mem_dsigmaxy_dy[p] + a_y[p]*dsigmaxydy
                mem_dsigmayy_dy_out[p] = b_y[p]*mem_dsigmayy_dy[p] + a_y[p]*dsigmayydy
                dsigmaxydy = (1./k_y[p])*dsigmaxydy + mem_dsigmaxy_dy_out[p]
                dsigmayydy = (1./k_y[p])*dsigmayydy + mem_dsigmayy_dy_out[p]

            #Cambiando las velocidades
            vxout[p] = vx[p]*(1-vel_damping) + dt*aa[p]*(dsigmaxxdx+dsigmaxydy)
            vyout[p] = vy[p]*(1-vel_damping) + dt*aa[p]*(dsigmaxydx+dsigmayydy)


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def init_memory_velocity(vx, vy, mem_dvx_dx, mem_dvy_dx, mem_dvx_dy, mem_dvy_dy,
                         k_x_half, k_y_half, pml_points_x1, pml_points_x2, pml_points_y1,
                         pml_points_y2, dx, dy, nx, ny):

    for j in prange(ny-1):
        en_y = j < pml_points_y1 or j >= ny-1-pml_points_y2
        for i in range(nx-1):
            en_x = i < pml_points_x1 or i >= nx-1-pml_points_x2
            q = i+(nx-1)*j
            if en_x:
                #Define derivadas para las velocidades:
                dvxdx = (0.5/dx)*(vx[i+1+nx*j]-vx[i+nx*j]+vx[i+1+nx*(j+1)]-vx[i+nx*(j+1)])
                dvydx = (0.5/dx)*(vy[i+1+nx*j]-vy[i+nx*j]+vy[i+1+nx*(j+1)]-vy[i+nx*(j+1)])
                #Cambiando la memoria de las derivadas de velocidad
                mem_dvx_dx[q] = (k_x_half[q]-1)*(1./k_x_half[q])*dvxdx
                mem_dvy_dx[q] = (k_x_half[q]-1)*(1./k_x_half[q])*dvydx
            if en_y:
                dvxdy = (0.5/dy)*(vx[i+nx*(j+1)]-vx[i+nx*j]+vx[i+1+nx*(j+1)]-vx[i+1+nx*j])
                dvydy = (0.5/dy)*(vy[i+nx*(j+1)]-vy[i+nx*j]+vy[i+1+nx*(j+1)]-vy[i+1+nx*j])
                mem_dvx_dy[q] = (k_y_half[q]-1.)*(1./k_y_half[q])*dvxdy
                mem_dvy_dy[q] = (k_y_half[q]-1.)*(1./k_y_half[q])*dvydy


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def init_memory_stress(sigmaxx, sigmaxy, sigmayy, mem_dsigmaxx_dx, mem_dsigmaxy_dx,
                       mem_dsigmaxy_dy, mem_dsigmayy_dy, k_x, k_y, pml_points_x1,
                       pml_points_x2, pml_points_y1, pml_points_y2, dx, dy, nx, ny):
    '''sigma* son los esfuerzos con borde (nx+1)*(ny+1).'''

    for j in prange(ny):
        en_y = j < pml_points_y1 or j >= ny-pml_points_y2
        for i in range(nx):
            en_x = i < pml_points_x1 or i >= nx-pml_points_x2
            p = i+nx*j
            if en_x:
                #Definiendo las derivadas
                dsigmaxxdx = (0.5/dx)*(sigmaxx[i+1+(nx+1)*j]-sigmaxx[i+(nx+1)*j]+sigmaxx[i+1+(nx+1)*(j+1)]-sigmaxx[i+(nx+1)*(j+1)])
                dsigmaxydx = (0.5/dx)*(sigmaxy[i+1+(nx+1)*j]-sigmaxy[i+(nx+1)*j]+sigmaxy[i+1+(nx+1)*(j+1)]-sigmaxy[i+(nx+1)*(j+1)])
                #Cambiando la memoria de las derivadas para los stresses
                mem_dsigmaxx_dx[p] = (k_x[p]-1)*(1./k_x[p])*dsigmaxxdx
                mem_dsigmaxy_dx[p] = (k_x[p]-1)*(1./k_x[p])*dsigmaxydx
            if en_y:
                dsigmaxydy = (0.5/dy)*(sigmaxy[i+(nx+1)*(j+1)]-sigmaxy[i+(nx+1)*j]+sigmaxy[i+1+(nx+1)*(j+1)]-sigmaxy[i+1+(nx+1)*j])
                dsigmayydy = (0.5/dy)*(sigmayy[i+(nx+1)*(j+1)]-sigmayy[i+(nx+1)*j]+sigmayy[i+1+(nx+1)*(j+1)]-sigmayy[i+1+(nx+1)*j])
                mem_dsigmaxy_dy[p] = (k_y[p]-1)*(1./k_y[p])*dsigmaxydy
                mem_dsigmayy_dy[p] = (k_y[p]-1)*(1./k_y[p])*dsigmayydy
