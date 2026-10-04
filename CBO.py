#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:22:57 2022

@author: claudiovenegas

Condiciones de borde oblicuas en las velocidades (superficie libre con
topografía), basadas en Jih et al. (1988).

Arreglos planos con índice i + nx*j y tamaño nx*ny: elevation_angle,
boundary_flag_velocity, vx_*, vy_*.
domain_flag (el arreglo "too" de TOP.py) es la excepción: tiene tamaño
(nx+1)*(ny+1) y se indexa i + (nx+1)*j.

IMPORTANTE: cboveljih LEE de vx_in/vy_in y ESCRIBE en vx_out/vy_out. Los
arreglos de entrada y de salida deben ser distintos (si no, los hilos se
pisan entre sí). Para trabajar sobre el mismo arreglo use cboveljih_inplace.
"""

import numpy as np
from numba import config, njit, prange, set_num_threads
from CF import vs, vp

# MacBook Air M4: 10 hilos. Si la máquina tiene menos, se usan los disponibles.
NUM_HILOS = 10
set_num_threads(min(NUM_HILOS, config.NUMBA_NUM_THREADS))

# Ángulos más pequeños que esto se consideran superficie plana (las fórmulas
# dividen por tan(angulo)).
ANGULO_MIN = 1e-9


@njit(fastmath=True, cache=True)
def _at(a, i, j, nx, ny):
    '''Lee a[i, j] acotando los índices a la malla (evita leer fuera del
    arreglo en nodos cercanos al borde o con pendientes muy empinadas).'''
    ii = min(max(np.int64(i), np.int64(0)), np.int64(nx-1))
    jj = min(max(np.int64(j), np.int64(0)), np.int64(ny-1))
    return a[ii+nx*jj]


@njit(fastmath=True, cache=True)
def _rot(vx, vy, ang):
    '''Rota el vector (vx, vy) en un ángulo ang.'''
    c = np.cos(ang)
    s = np.sin(ang)
    return vx*c - vy*s, vx*s + vy*c


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def cboveljih(elevation_angle, boundary_flag_velocity, domain_flag, vx_in, vy_in, vx_out, vy_out, vp, vs, dx, dy, nx, ny):
    # (lambda/(lambda+2mu)) = 1 - 2 (vs/vp)^2
    poisson = 1. - 2.*(vs/vp)**2

    #Space loop
    for j in prange(ny):
        for i in range(nx):
            #Read domain flag
            domain = domain_flag[i+(nx+1)*j]

            '''Hay tres casos: encima de la línea, en la línea y debajo de la
            línea. Cuando está por encima de la línea, todas las velocidades
            son cero. Cuando esté debajo de la línea, no haga nada, y
            cuando esté en la línea, haga el truco. '''

            if domain > 1:
                vx_out[i+nx*j] = 0.
                vy_out[i+nx*j] = 0.

            ''' Sobre la linea'''
            if abs(domain-1) < 0.1:
                '''Aquí estamos, en la línea. Ahora es el momento
                de leer más valores'''
                '''angulo de elevación. Esta matriz está compuesta por una
                columna con un solo valor (el mismo valor en todos los
                elementos de la columna) '''
                el_angle = abs(elevation_angle[i+nx*j])
                '''Bandera de límite (tipo de transición)'''
                bc_type = boundary_flag_velocity[i+nx*j]

                ''' Ahora se necesita aplicar condiciones basadas en el tipo
                de transición. Esta considera una rotación inicial del dominio,
                luego una solución y finalmente una rotación de regreso.'''

                ''' Caso 1: Pendiente empinada constante'''
                if abs(bc_type - 1.) < 0.1:
                    #Definir el epsilon
                    epsilon = np.sin(el_angle)*np.sin(el_angle)
                    '''Los cuatro puntos en los que necesitamos rotar
                    velocidades aquí. Son (i, j), (i+1, j), (i, j+m) y (0,0).
                    Aquí, (0,0) es un punto donde podemos interpolar las
                    velocidades, basándonos en el otro.

                    Necesito hacer lo mismo con (i, j+m), ya que m es un valor
                    de pendiente y, por lo tanto, no un numero entero.
                    jm = (int) (j tan (el_angle))'''
                    jm = int(j+np.tan(el_angle))

                    vx_ijm = _at(vx_in, i, jm, nx, ny)
                    vy_ijm = _at(vy_in, i, jm, nx, ny)
                    vx_i1j = _at(vx_in, i+1, j, nx, ny)
                    vy_i1j = _at(vy_in, i+1, j, nx, ny)
                    vx_i1j1 = _at(vx_in, i+1, j-1, nx, ny)
                    vy_i1j1 = _at(vy_in, i+1, j-1, nx, ny)
                    vx0 = epsilon*vx_i1j + (1-epsilon)*vx_ijm
                    vy0 = epsilon*vy_i1j + (1-epsilon)*vy_ijm
                    vx_ijm1 = _at(vx_in, i, jm-1, nx, ny)
                    vy_ijm1 = _at(vy_in, i, jm-1, nx, ny)

                    ''' Ahora necesitamos rotar las velocidades en dos puntos'''
                    vx0_rot, vy0_rot = _rot(vx0, vy0, -el_angle)
                    vx_i1j1_rot, vy_i1j1_rot = _rot(vx_i1j1, vy_i1j1, -el_angle)
                    vx_ijm1_rot, vy_ijm1_rot = _rot(vx_ijm1, vy_ijm1, -el_angle)
                    ''' La solución del problema esta dado por  (vx_ij_rot, vy_ij_rot)'''
                    vx_ij_rot = vx0_rot - np.cos(el_angle)*np.sin(el_angle)*(vy_i1j1_rot - vy_ijm1_rot)
                    vy_ij_rot = vy0_rot - np.cos(el_angle)*np.sin(el_angle)*poisson*(vx_i1j1_rot - vx_ijm1_rot)

                    '''Ahora, debemos realizar la rotación inversa'''
                    vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, el_angle)


                '''Caso 2: Pendiente suave constante'''
                if abs(bc_type - 2.) < 0.1 and el_angle > ANGULO_MIN:
                    #Definir epsilon
                    epsilon = np.sin(el_angle)*np.sin(el_angle)
                    '''Los cuatro puntos en los que necesitamos rotar
                    velocidades aquí. Son (i, j), (i+1, j), (i, j+m) y (0,0).
                    Aquí, (0,0) es un punto donde podemos interpolar las
                    velocidades, basándonos en el otro.

                    Necesito hacer lo mismo con (i, j+m), ya que m es un valor
                    de pendiente y, por lo tanto, no un numero entero.'''

                    vx_ijm = _at(vx_in, i, j+1, nx, ny)
                    vy_ijm = _at(vy_in, i, j+1, nx, ny)
                    vx_i1j = _at(vx_in, i+1, j, nx, ny)
                    vy_i1j = _at(vy_in, i+1, j, nx, ny)
                    vx0 = epsilon*vx_i1j + (1-epsilon)*vx_ijm
                    vy0 = epsilon*vy_i1j + (1-epsilon)*vy_ijm

                    '''Ahora necesito rotar velocidades en dos puntos.'''
                    vx0_rot, vy0_rot = _rot(vx0, vy0, -el_angle)
                    vx_i1j_rot, vy_i1j_rot = _rot(vx_i1j, vy_i1j, -el_angle)
                    vx_ijm_rot, vy_ijm_rot = _rot(vx_ijm, vy_ijm, -el_angle)
                    ''' La solución del problema esta dado por  (vx_ij_rot, vy_ij_rot)'''
                    vx_ij_rot = vx0_rot - np.cos(el_angle)*np.sin(el_angle)*(vy_i1j_rot - vy_ijm_rot)
                    vy_ij_rot = vy0_rot - np.cos(el_angle)*np.sin(el_angle)*poisson*(vx_i1j_rot - vx_ijm_rot)
                    '''Ahora, debemos realizar la rotación inversa'''
                    vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, el_angle)

                '''Caso 3: Transición cóncava horizontal a pendiente suave'''
                if abs(bc_type - 3.) < 0.1 and el_angle > ANGULO_MIN:
                    '''Estamos considerando los valores de pendiente en el elemento
                    izquierdo de la malla. Aquí, el ángulo de elevación se define
                    en el elemento derecho de la malla. Por lo tanto, por lo que
                    es necesario realizar un cambio.'''
                    '''Actualización: aparentemente, y dado que hay un problema
                    con la forma en que estoy asignando valores a la matriz
                    elevación_angle, estoy intentando ahora con el valor que
                    corresponde.'''
                    #Definiendo los epsilon
                    epsilon0 = np.tan(0.5*el_angle)
                    epsilon1 = np.tan(0.5*el_angle)/np.tan(el_angle)
                    '''Los cuatro puntos en los que necesitamos rotar
                    velocidades aquí. Son (i, j), (i+1, j), (i, j+m) y (0,0).
                    Aquí, (0,0) es un punto donde podemos interpolar las
                    velocidades, basándonos en el otro.

                    Necesito hacer lo mismo con (i, j+m), ya que m es un valor
                    de pendiente y, por lo tanto, no un numero entero.'''

                    jmp = int(j+np.tan(el_angle))
                    jmm = int(j-np.tan(el_angle))
                    vx_ij1 = _at(vx_in, i, j+1, nx, ny)
                    vy_ij1 = _at(vy_in, i, j+1, nx, ny)
                    vx_i1j1 = _at(vx_in, i+1, j+1, nx, ny)
                    vy_i1j1 = _at(vy_in, i+1, j+1, nx, ny)
                    vx_im1j1 = _at(vx_in, i-1, j+1, nx, ny)
                    vy_im1j1 = _at(vy_in, i-1, j+1, nx, ny)
                    vx_im1jmp1 = _at(vx_in, i-1, jmp+1, nx, ny)
                    vy_im1jmp1 = _at(vy_in, i-1, jmp+1, nx, ny)
                    vx_ip1jmm1 = _at(vx_in, i+1, jmm+1, nx, ny)
                    vy_ip1jmm1 = _at(vy_in, i+1, jmm+1, nx, ny)
                    #Definiendo los puntos en los que realizaremos la rotación
                    vx0 = (1-epsilon0)*vx_ij1 + epsilon0*vx_i1j1
                    vy0 = (1-epsilon0)*vy_ij1 + epsilon0*vy_i1j1
                    vx1 = (1-epsilon1)*vx_im1jmp1 + epsilon1*vx_im1j1
                    vy1 = (1-epsilon1)*vy_im1jmp1 + epsilon1*vy_im1j1
                    vx2 = (1-epsilon1)*vx_ip1jmm1 + epsilon1*vx_i1j1
                    vy2 = (1-epsilon1)*vy_ip1jmm1 + epsilon1*vy_i1j1

                    '''Ahora debemos realizar una rotación de velocidades
                    en varios puntos'''
                    vx0_rot, vy0_rot = _rot(vx0, vy0, -0.5*el_angle)
                    vx1_rot, vy1_rot = _rot(vx1, vy1, -0.5*el_angle)
                    vx2_rot, vy2_rot = _rot(vx2, vy2, -0.5*el_angle)

                    '''A continuación, se encuentra la solución del problema
                    (vx_ij_rot, vy_ij_rot)'''
                    vx_ij_rot = vx0_rot - 0.5*(dy/dx)*(vy2_rot - vy1_rot)
                    vy_ij_rot = vy0_rot - 0.5*(dy/dx)*poisson*(vx2_rot - vx1_rot)

                    '''Ahora realizaremos la rotación inversa'''
                    vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, 0.5*el_angle)

                '''Caso 4: Tránsito cóncavo de pendiente suave a empinada'''
                if abs(bc_type - 4.) < 0.1:
                    '''Aquí pasamos de una pendiente ndy / dz a la izquierda
                    a mdy / dx a la derecha. Primero, necesito considerar ambos
                    ángulos de elevación, izquierda y derecha.'''
                    el_angle_left = abs(elevation_angle[i+nx*j])
                    el_angle_right = abs(_at(elevation_angle, i+1, j, nx, ny))
                    el_angle_aux = 0.5*(el_angle_left+el_angle_right)
                    if el_angle_aux > ANGULO_MIN:
                        #Definimos los epsilon
                        epsilon0 = np.tan(el_angle_left)/np.tan(el_angle_aux)
                        epsilon1 = epsilon0
                        '''Definiremos las velocidades que se necesitaran más adelante'''
                        jpn = int(j+np.tan(el_angle_left))
                        jmn = int(j-np.tan(el_angle_right))
                        vx_i1j = _at(vx_in, i+1, j, nx, ny)
                        vy_i1j = _at(vy_in, i+1, j, nx, ny)
                        vx_i1jpn = _at(vx_in, i+1, jpn, nx, ny)
                        vy_i1jpn = _at(vy_in, i+1, jpn, nx, ny)
                        vx_ijpn = _at(vx_in, i, jpn, nx, ny)
                        vy_ijpn = _at(vy_in, i, jpn, nx, ny)
                        vx_im1jpn = _at(vx_in, i-1, jpn, nx, ny)
                        vy_im1jpn = _at(vy_in, i-1, jpn, nx, ny)
                        vx_i1jmn = _at(vx_in, i+1, jmn, nx, ny)
                        vy_i1jmn = _at(vy_in, i+1, jmn, nx, ny)
                        '''Defina las velocidades en ciertos puntos.
                        Se rotarán más tarde.'''
                        vx0 = (1-epsilon0)*vx_i1j + epsilon0*vx_i1jpn
                        vy0 = (1-epsilon0)*vy_i1j + epsilon0*vy_i1jpn
                        vx1 = (1-epsilon1)*vx_ijpn + epsilon1*vx_im1jpn
                        vy1 = (1-epsilon1)*vy_ijpn + epsilon1*vy_im1jpn
                        vx2 = (1-epsilon1)*vx_i1j + epsilon1*vx_i1jmn
                        vy2 = (1-epsilon1)*vy_i1j + epsilon1*vy_i1jmn
                        '''Ahora necesitamos realizar la rotación'''
                        vx0_rot, vy0_rot = _rot(vx0, vy0, -el_angle_aux)
                        vx1_rot, vy1_rot = _rot(vx1, vy1, -el_angle_aux)
                        vx2_rot, vy2_rot = _rot(vx2, vy2, -el_angle_aux)
                        '''Ahora la solución en este problema esta dado por:
                            (vx_ij_rot, vy_ij_rot)'''
                        fac = 1./(np.tan(el_angle_aux)+np.tan(el_angle_left))
                        vx_ij_rot = vx0_rot - fac*(vy2_rot - vy1_rot)
                        vy_ij_rot = vy0_rot - fac*poisson*(vx2_rot - vx1_rot)
                        '''Ahora rotaremos para regresar'''
                        vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, el_angle_aux)

                '''Caso 5: Convex change in slope'''
                if abs(bc_type - 5.) < 0.1:
                    '''Esta vez nos movemos de una pendiente mdy/dx a la
                    izquierda a ndy/dx a la derecha. ¡Primero, ángulos de
                    elevación!'''
                    el_angle_left = abs(elevation_angle[i+nx*j])
                    el_angle_right = abs(_at(elevation_angle, i+1, j, nx, ny))
                    el_angle_aux = 0.5*(el_angle_left+el_angle_right)

                    if el_angle_aux > ANGULO_MIN and el_angle_left > ANGULO_MIN:
                        #Definiendo los epsilon
                        #epsilon0
                        epsilon0 = np.tan(el_angle_aux)
                        #epsilon2, que necesitaremos más tarde
                        epsilon2 = 0.
                        #Definimos las velocidades importantes
                        vx_ij1 = _at(vx_in, i, j+1, nx, ny)
                        vy_ij1 = _at(vy_in, i, j+1, nx, ny)
                        vx_i1j1 = _at(vx_in, i+1, j+1, nx, ny)
                        vy_i1j1 = _at(vy_in, i+1, j+1, nx, ny)
                        vx_i1j = _at(vx_in, i+1, j, nx, ny)
                        vy_i1j = _at(vy_in, i+1, j, nx, ny)

                        #Iniciando las velocidades importantes.
                        vx0 = 0.
                        vy0 = 0.
                        '''Ahora tengo dos casos: cuando epsilon0 es menor que uno,
                        o bien mayor o igual que uno.'''
                        if epsilon0 < 1:
                            vx0 = (1-epsilon0)*vx_ij1 + epsilon0*vx_i1j1
                            vy0 = (1-epsilon0)*vy_ij1 + epsilon0*vy_i1j1
                            epsilon2 = np.tan(el_angle_left)

                        else:
                            epsilon3 = 1/epsilon0
                            vx0 = (1-epsilon3)*vx_i1j1 + epsilon3*vx_i1j
                            vy0 = (1-epsilon3)*vy_i1j1 + epsilon3*vy_i1j
                            epsilon2 = 1/np.tan(el_angle_aux)

                        '''Ahora, dependiendo del valor de m, tenemos dos casos.
                        Así que primero definamos m'''
                        m = np.tan(el_angle_left)
                        #caso 1: m =1 (comparación con tolerancia: tan(45°) no es exactamente 1)
                        if abs(m-1.) < 1e-6:
                            #Otra velocidad importante
                            jpn = int(j+np.tan(el_angle_right))
                            vx_ijpn = _at(vx_in, i, jpn, nx, ny)
                            vy_ijpn = _at(vy_in, i, jpn, nx, ny)
                            #Otro epsilon
                            epsilon1 = (np.tan(el_angle_aux)/m-1)/np.tan(el_angle_left)
                            #Luego
                            vx1 = (1-epsilon1)*vx_ij1 + epsilon1*vx_ijpn
                            vy1 = (1-epsilon1)*vy_ij1 + epsilon1*vy_ijpn
                            #Rontando en puntos imporantes
                            vx0_rot, vy0_rot = _rot(vx0, vy0, -el_angle_aux)
                            vx_i1j_rot, vy_i1j_rot = _rot(vx_i1j, vy_i1j, -el_angle_aux)
                            vx1_rot, vy1_rot = _rot(vx1, vy1, -el_angle_aux)
                            #La solución al problema es:
                            vx_ij_rot = vx0_rot - epsilon2*(vy_i1j_rot - vy1_rot)
                            vy_ij_rot = vy0_rot - epsilon2*poisson*(vx_i1j_rot - vx1_rot)
                            #Regresando la rotación original
                            vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, el_angle_aux)

                        elif m > 1:
                            epsilon1 = (2.*dy)/(dx*np.tan(el_angle_aux))
                            #Velocidades importantes
                            vx_ijm1 = _at(vx_in, i, j+1, nx, ny)
                            vy_ijm1 = _at(vy_in, i, j+1, nx, ny)
                            vx_i1jm1 = _at(vx_in, i+1, j-1, nx, ny)
                            vy_i1jm1 = _at(vy_in, i+1, j-1, nx, ny)
                            #Luego
                            vx1 = (1-epsilon1)*vx_ijm1 + epsilon1*vx_i1jm1
                            vy1 = (1-epsilon1)*vy_ijm1 + epsilon1*vy_i1jm1
                            #Rotando
                            vx0_rot, vy0_rot = _rot(vx0, vy0, -el_angle_aux)
                            vx_ij1_rot, vy_ij1_rot = _rot(vx_ij1, vy_ij1, -el_angle_aux)
                            vx1_rot, vy1_rot = _rot(vx1, vy1, -el_angle_aux)

                            if epsilon0 <= 1:
                                epsilon2 = 0.5*np.tan(el_angle_aux)
                            else:
                                epsilon2 = 0.5*dy/dx

                            #La solución del problema:
                            vx_ij_rot = vx0_rot - epsilon2*(vy1_rot - vy_ij1_rot)
                            vy_ij_rot = vy0_rot - epsilon2*poisson*(vx1_rot - vx_ij1_rot)

                            #Regresando la rotación:
                            vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, el_angle_aux)
                        '''Si m < 1 no hay fórmula definida en este código: el nodo
                        conserva el valor que trae vx_out.'''


                '''Caso 6: Transición suave convexa de pendiente a horizontal'''
                if abs(bc_type - 6.) < 0.1:
                    '''Ángulo de elevación, del lado izquierdo de la transición
                    (el lado derecho no tiene elevación, ¡es plano!)'''
                    #Actualización: probando con i-1
                    el_angle = abs(_at(elevation_angle, i-1, j, nx, ny))
                    if el_angle > ANGULO_MIN:
                        #Epsilons
                        epsilon0 = np.tan(0.5*el_angle)*np.tan(el_angle)
                        epsilon1 = np.tan(0.5*el_angle)/np.tan(el_angle)

                        #Velocidades importantes
                        vx_ij1 = _at(vx_in, i, j+1, nx, ny)
                        vy_ij1 = _at(vy_in, i, j+1, nx, ny)
                        vx_i1j1 = _at(vx_in, i+1, j+1, nx, ny)
                        vy_i1j1 = _at(vy_in, i+1, j+1, nx, ny)
                        vx_im1j1 = _at(vx_in, i-1, j+1, nx, ny)
                        vy_im1j1 = _at(vy_in, i-1, j+1, nx, ny)
                        vx_im1j2 = _at(vx_in, i-1, j+2, nx, ny)
                        vy_im1j2 = _at(vy_in, i-1, j+2, nx, ny)
                        vx_i1j = _at(vx_in, i+1, j, nx, ny)
                        vy_i1j = _at(vy_in, i+1, j, nx, ny)
                        #Calculando la interpolación de velocidades
                        vx0 = (1-epsilon0)*vx_ij1 + epsilon0*vx_i1j1
                        vy0 = (1-epsilon0)*vy_ij1 + epsilon0*vy_i1j1
                        vx1 = (1-epsilon1)*vx_im1j1 + epsilon1*vx_im1j2
                        vy1 = (1-epsilon1)*vy_im1j1 + epsilon1*vy_im1j2
                        vx2 = (1-epsilon1)*vx_i1j1 + epsilon1*vx_i1j
                        vy2 = (1-epsilon1)*vy_i1j1 + epsilon1*vy_i1j
                        #Rotando las vellocidades importantes
                        vx0_rot, vy0_rot = _rot(vx0, vy0, -0.5*el_angle)
                        vx1_rot, vy1_rot = _rot(vx1, vy1, -0.5*el_angle)
                        vx2_rot, vy2_rot = _rot(vx2, vy2, -0.5*el_angle)
                        #La solución del problema
                        vx_ij_rot = vx0_rot - (0.5*np.tan(el_angle))*(vy2_rot - vy1_rot)
                        vy_ij_rot = vy0_rot - (0.5*np.tan(el_angle))*poisson*(vx2_rot - vx1_rot)
                        #Retrocediendo la rotación
                        vx_out[i+nx*j], vy_out[i+nx*j] = _rot(vx_ij_rot, vy_ij_rot, 0.5*el_angle)

            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                vx_out[i+nx*j] = 0.
                vy_out[i+nx*j] = 0.


def cboveljih_inplace(elevation_angle, boundary_flag_velocity, domain_flag, vx, vy, vp, vs, dx, dy, nx, ny):
    '''Aplica cboveljih modificando vx, vy directamente. Usa una copia de las
    velocidades como entrada para que los hilos no lean valores ya modificados.'''
    vx_copia = vx.copy()
    vy_copia = vy.copy()
    cboveljih(elevation_angle, boundary_flag_velocity, domain_flag, vx_copia, vy_copia, vx, vy, vp, vs, dx, dy, nx, ny)


# vx_out = np.zeros(nx*ny)
# vy_out = np.zeros(nx*ny)
# vx_in = 3*np.ones(nx*ny)
# vy_in = 4*np.ones(nx*ny)
# cboveljih(elevation_values, boundary_flag_velocity, to, vx_in, vy_in, vx_out, vy_out, vp, vs, dx, dy, nx, ny)
