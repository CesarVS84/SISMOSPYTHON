# -*- coding: utf-8 -*-
"""
CPMLD.py: núcleos de la CPML (Convolutional PML) para elastodinámica 2-D.

Estas funciones actualizan SOLO las celdas/nodos que están dentro de la capa y
sirven de referencia (y para pruebas) del núcleo fusionado que usa CPML.py
(paso_velocidad / paso_esfuerzo), que hace lo mismo pero junto con la
actualización del interior en una sola pasada.

Convenciones de memoria (todas con índice plano i + nx*j):
    - velocidades y perfiles "en nodo"      : nx*ny
    - esfuerzos interiores y perfiles "half": (nx-1)*(ny-1)  (celda i entre los
                                              nodos i e i+1)
    - esfuerzos con borde (Sbxx, Sbxy, Sbyy): (nx+1)*(ny+1)  (Sb[i] = celda i-1)

b, c, d (cpml_stress) son (lambda+2mu), lambda y mu y aa (cpml_vel) es 1/rho,
SIN dividir por dx: estas funciones ya dividen por dx/dy al derivar.

Cada función recorre la malla una sola vez en paralelo (un hilo por fila).
Una celda (o nodo) puede estar en la capa en x, en y, o en ambas (esquina): se
actualiza la memoria de la(s) derivada(s) que corresponda, una sola vez.
"""

import numpy as np
from numba import njit, prange
from CBA import damp_profiles, crear_perfiles


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
