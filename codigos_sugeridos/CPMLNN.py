# -*- coding: utf-8 -*-
"""
CPMLNN.py: CPML con densidad ESCALAR (medio homogéneo).

Misma interfaz que el CPMLNN.py original: cpml_vel recibe Rho (escalar) en lugar del
arreglo aa = 1/rho. Todo lo demás es CPMLD.py (no se duplica código).
"""

import numpy as np
from CBA import damp_profiles, crear_perfiles
from CPMLD import cpml_stress, init_memory_velocity, init_memory_stress
from CPMLD import cpml_vel as _cpml_vel


def cpml_vel(vxout, vyout, sigmaxx, sigmaxy, sigmayy, vx, vy, mem_dsigmaxx_dx,
             mem_dsigmaxy_dx, mem_dsigmaxy_dy, mem_dsigmayy_dy, mem_dsigmaxx_dx_out,
             mem_dsigmaxy_dx_out, mem_dsigmaxy_dy_out, mem_dsigmayy_dy_out,
             a_x, a_y, k_x, alpha_x, b_x, k_y, alpha_y, b_y, pml_points_x1,
             pml_points_x2, pml_points_y1, pml_points_y2, Rho, dx, dy, dt,
             vel_damping, nx, ny):
    aa = np.full(nx*ny, 1./Rho)
    _cpml_vel(vxout, vyout, sigmaxx, sigmaxy, sigmayy, vx, vy, mem_dsigmaxx_dx,
              mem_dsigmaxy_dx, mem_dsigmaxy_dy, mem_dsigmayy_dy, mem_dsigmaxx_dx_out,
              mem_dsigmaxy_dx_out, mem_dsigmaxy_dy_out, mem_dsigmayy_dy_out,
              a_x, a_y, k_x, alpha_x, b_x, k_y, alpha_y, b_y, pml_points_x1,
              pml_points_x2, pml_points_y1, pml_points_y2, aa, dx, dy, dt,
              vel_damping, nx, ny)
