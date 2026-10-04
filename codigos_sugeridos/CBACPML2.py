# -*- coding: utf-8 -*-
"""
CBACPML2.py: CPML con la fuente aplicada dentro de cpml_vel (nodos F == 50 / 90).

Misma interfaz que el CBACPML2.py original: cpml_vel(..., r, theta, t). r debe ser una
función de t (en este paquete, fuente.r). El programa principal (CPML.py) aplica la
fuente antes de avanzar la velocidad; esta variante la impone después, sobre las
velocidades nuevas, como hacía el original.
"""

import numpy as np
from CBA import damp_profiles, crear_perfiles
from CPMLD import cpml_stress, init_memory_velocity, init_memory_stress
from CPMLD import cpml_vel as _cpml_vel
import TOP


def cpml_vel(vxout, vyout, sigmaxx, sigmaxy, sigmayy, vx, vy, mem_dsigmaxx_dx,
             mem_dsigmaxy_dx, mem_dsigmaxy_dy, mem_dsigmayy_dy, mem_dsigmaxx_dx_out,
             mem_dsigmaxy_dx_out, mem_dsigmaxy_dy_out, mem_dsigmayy_dy_out,
             a_x, a_y, k_x, alpha_x, b_x, k_y, alpha_y, b_y, pml_points_x1,
             pml_points_x2, pml_points_y1, pml_points_y2, aa, dx, dy, dt,
             vel_damping, nx, ny, r, theta, t, F=TOP.F):
    _cpml_vel(vxout, vyout, sigmaxx, sigmaxy, sigmayy, vx, vy, mem_dsigmaxx_dx,
              mem_dsigmaxy_dx, mem_dsigmaxy_dy, mem_dsigmayy_dy, mem_dsigmaxx_dx_out,
              mem_dsigmaxy_dx_out, mem_dsigmaxy_dy_out, mem_dsigmayy_dy_out,
              a_x, a_y, k_x, alpha_x, b_x, k_y, alpha_y, b_y, pml_points_x1,
              pml_points_x2, pml_points_y1, pml_points_y2, aa, dx, dy, dt,
              vel_damping, nx, ny)
    rt = r(t)
    for valor, signo in ((50, -1.), (90, 1.)):
        for i, j in np.argwhere(np.asarray(F) == valor):
            if 0 < i < nx-1 and 0 < j < ny-1:
                vxout[i+nx*j] = signo*0.5*rt*np.sin(theta)
                vyout[i+nx*j] = -signo*0.5*rt*np.cos(theta)
