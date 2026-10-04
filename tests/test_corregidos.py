"""Pruebas de los archivos corregidos de la carpeta raíz (CF, CPMLD, CBO)."""
import numpy as np
import CF
import CPMLD
import CBO


def test_cfl_exacto():
    assert CF.dt < CF.dtestable


def test_perfiles_cpml_simetricos():
    nx, ny, P = CF.nx, CF.ny, CPMLD.pml_points_x1
    dxn = CPMLD.d_x.reshape(ny, nx)[ny//2]
    dxh = CPMLD.d_x_half.reshape(ny-1, nx-1)[ny//2]
    dyn = CPMLD.d_y.reshape(ny, nx)[:, nx//2]
    assert np.allclose(dxn, dxn[::-1]) and np.allclose(dxh, dxh[::-1]) and np.allclose(dyn, dyn[::-1])
    # d0 con logaritmo natural
    assert abs(CPMLD.d0x + 3*CPMLD.cp*np.log(CPMLD.Rcoef)/(2*P*CF.dx)) < 1e-9


def test_cbo_tamano_real_de_too():
    nx = ny = 40
    dx = dy = 50.
    vp, vs = 5669., 3273.
    for tipo in range(1, 7):
        dom = np.zeros((nx+1)*(ny+1))              # too: (nx+1)*(ny+1), i + (nx+1)*j
        ang = np.zeros(nx*ny)
        flg = np.zeros(nx*ny)
        dom[20+(nx+1)*20] = 1
        dom[:(nx+1)*5] = 2                         # aire en las 5 primeras filas (j pequeño aquí)
        flg[20+nx*20] = tipo
        ang[:] = np.radians(40.)
        vx = np.full(nx*ny, 3.)
        vy = np.full(nx*ny, 4.)
        vxo = np.full(nx*ny, -999.)
        vyo = np.full(nx*ny, -999.)
        CBO.cboveljih(ang, flg, dom, vx, vy, vxo, vyo, vp, vs, dx, dy, nx, ny)
        if vxo[20+nx*20] != -999.:
            assert abs(vxo[20+nx*20]-3) < 1e-9 and abs(vyo[20+nx*20]-4) < 1e-9
        assert np.all(vxo[:nx*5] == 0.)            # aire: velocidad cero
