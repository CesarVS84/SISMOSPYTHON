#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:23:28 2021

@author: claudiovenegas
"""

'''Condiciones de bordes oblicuas en las velocidades. Basadas en Jih
en el año 1988'''

'''Esta fue la primera versión (con guvectorize). Ya no contiene su propia
implementación: la versión vigente y corregida está en CBO.py (cboveljih),
que recibe todos los arreglos por argumento. Aquí se conserva la interfaz
antigua, cboveljih(vx_in, vy_in, vx_out, vy_out), que toma la topografía de
prueba.py y TOP.py y delega en CBO.cboveljih.

Diferencias respecto de la versión antigua: vs y vp vienen de CF.py (antes
estaban escritos a mano como 200 y 4500, distintos de los del modelo), los
arreglos son de tamaño nx*ny y ya no se ejecuta nada al importar el módulo.'''

import numpy as np
from CBO import cboveljih as _cboveljih
from CF import vs, vp, dx, dy, nx, ny


def cboveljih(vx_in, vy_in, vx_out, vy_out):
    # La topografía se importa al usarla (no al importar este módulo)
    from prueba import elevation_values, boundary_flag_velocity
    from TOP import to
    _cboveljih(elevation_values, boundary_flag_velocity, to, vx_in, vy_in,
               vx_out, vy_out, vp, vs, dx, dy, nx, ny)
