#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:24:52 2022

@author: claudiovenegas

Pendientes de la superficie y tipo de transición (banderas) para las
condiciones de borde de Jih (CBO.py). Versión revisada.

    boundary_flag_velocity (nx*ny): tipo de transición 1..6 en el nodo central
    elevation_values       (nx*ny): ángulo (rad) del tramo que va del nodo i al
                                    i+1, guardado en toda la columna i

Cambios respecto del original: la clasificación se hace UNA vez por nodo, cuando
los tres nodos vecinos (izquierdo, central y derecho) ya fueron encontrados. En
el original ese bloque se ejecutaba en cada fila k, también antes de encontrar
los vecinos, con tan_left y tan_right de la iteración anterior (sin inicializar
en la primera), lo que podía dejar banderas incorrectas y escribir en la columna 0.
Además ya no se lee más allá de la última columna.
"""

import numpy as np
from numba import config, njit, prange, set_num_threads
from TOP import too
from CF import nx, ny, dx, dy

# MacBook Air M4: 10 hilos. Si la máquina tiene menos, se usan los disponibles.
set_num_threads(min(10, config.NUMBA_NUM_THREADS))

#inicializar
boundary_flag_velocity = np.zeros((nx)*(ny))
elevation_values = np.zeros((nx)*(ny))


@njit()
def identificador_de_pendientes(bc_flag, boundary_flag_velocity, elevation_values):
    # Fila de la superficie (bandera 1) en cada columna de bc_flag, que tiene
    # tamaño (nx+1)*(ny+1). 0 = no se encontró (el original tampoco usa la fila 0).
    fila = np.zeros(nx+1, dtype=np.int64)
    for i in range(nx+1):
        for k in range(ny):
            if bc_flag[i+(nx+1)*k] == 1:
                fila[i] = k

    # Tríos de columnas (i, i+1, i+2). El original no usa la columna 0.
    for i in range(1, nx-1):
        left_k = fila[i]
        center_k = fila[i+1]
        right_k = fila[i+2]
        if left_k == 0 or center_k == 0 or right_k == 0:
            continue
        center_i = i+1

        up_diff_left = -(center_k - left_k)*dy
        down_diff_left = dx
        tan_left = up_diff_left/down_diff_left

        up_diff_right = -(right_k - center_k)*dy
        down_diff_right = dx
        tan_right = up_diff_right/down_diff_right

        #Mini bucle espacial: el ángulo vale lo mismo en toda la columna
        for m in range(ny):
            elevation_values[i+nx*m] = np.arctan(tan_left)

        #Ahora puedo identificar los diferentes tipos de pendientes:
        #Primero necesitamos una diferencia de pendientes:
        #diferencias de pendientes aplicados en una montaña
        slope_diff = tan_right - tan_left

        #Ahora necesitamos reconocer los tipos de transición
        #1: slope_diff = 0, and tan_left >=2:
        if (slope_diff == 0. and tan_left >= 2.):
            #Pendiente empinada constante
            boundary_flag_velocity[center_i + nx*center_k] = 1.

        #2: slope_diff = 0 and tan_left <2
        if (slope_diff == 0. and tan_left < 2.):
            #Pendiente suave constante
            boundary_flag_velocity[center_i + nx*center_k] = 2.

        #3: slope_diff >0, tan_left = 0, and tan right >=1
        if slope_diff > 0. and tan_left == 0. and tan_right >= 1.:
            #Transición cóncava horizontal a pendiente suave
            boundary_flag_velocity[center_i + nx*center_k] = 3.

        #4: slope_diff >0, tan_left < tan_right
        if slope_diff > 0. and tan_left < tan_right and tan_left != 0.:
            #Transición cóncava de pendiente suave a empinada
            boundary_flag_velocity[center_i + nx*center_k] = 4.

        #5: slope_diff <0 , tan_right >0
        if slope_diff < 0. and tan_right > 0.:
            #Cambio convexo de pendiente
            boundary_flag_velocity[center_i + nx*center_k] = 5.

        #6: slope_diff <0, tan_right =0
        if slope_diff < 0. and tan_right == 0.:
            #Pendiente suave convexa a transición horizontal
            boundary_flag_velocity[center_i + nx*center_k] = 6.

    boundary_flag_velocity[0] = 0


identificador_de_pendientes(too, boundary_flag_velocity, elevation_values)


W = np.reshape(boundary_flag_velocity,(nx,ny))
bound = np.reshape(W,nx*ny)
Z = np.reshape(elevation_values,(nx,ny))
elevation = np.reshape(Z,nx*ny)
# Condiciones de Jih. En el original estas dos líneas invertían los arreglos planos:
#     boundary_flag_velocity = boundary_flag_velocity[::-1]
#     elevation = elevation[::-1]
# Eso refleja el campo tanto en i como en j (giro de 180 grados), mientras que "too"
# (bandera de dominio) no se invertía. Las banderas se calculan sobre los nodos de
# "too" que valen 1, de modo que al invertirlas casi ninguna queda sobre la superficie
# (medido: 0 de 500 nodos), y CBO.py solo corrige los nodos de la superficie: en la
# práctica las condiciones de Jih NO se aplicaban. Además, al activarlas sin
# invertir, la simulación se vuelve inestable (la amplitud crece 10^6 veces en 1000
# pasos), porque las fórmulas de CBO.py suponen el sólido hacia j creciente y este
# modelo tiene j hacia arriba (ver codigos_sugeridos/CBO.py, que acepta el reflejo).
# Por eso se deja explícito: con ACTIVAR_JIH = False no hay corrección de borde.
ACTIVAR_JIH = False
if not ACTIVAR_JIH:
    boundary_flag_velocity = np.zeros(nx*ny)
    elevation = np.zeros(nx*ny)
