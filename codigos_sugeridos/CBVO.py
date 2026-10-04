# -*- coding: utf-8 -*-
"""
CBVO.py: primera versión de las condiciones de borde oblicuas (Jih, 1988).

Ya no contiene su propia implementación (la original, con guvectorize, no
funcionaba): conserva la interfaz antigua, cboveljih(vx_in, vy_in, vx_out, vy_out),
y delega en CBO.cboveljih con la topografía de TOP.py / prueba.py.
"""

from CBO import cboveljih as _cboveljih
from CF import vs, vp, dx, dy, nx, ny


def cboveljih(vx_in, vy_in, vx_out, vy_out):
    import TOP
    import prueba
    for elevation, flag, si, sj in ((prueba.elevation, prueba.boundary_flag_velocity, 1, -1),
                                    (prueba.elevation_izq, prueba.boundary_flag_velocity_izq, -1, -1)):
        _cboveljih(elevation, flag, TOP.too, vx_in, vy_in, vx_out, vy_out, vp, vs, dx, dy, nx, ny, si, sj)
