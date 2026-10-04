"""CBO: un campo uniforme no cambia, y el kernel reflejado es equivalente."""
import numpy as np
import CBO

nx = ny = 40
dx = dy = 50.
vp, vs = 5669., 3273.


def modelo(tipo, grados, i0=20, j0=20):
    dom = np.zeros((ny+1, nx+1))          # (nx+1)*(ny+1), indexado i + (nx+1)*j
    ang = np.zeros((ny, nx))
    flg = np.zeros((ny, nx))
    dom[j0, i0] = 1
    flg[j0, i0] = tipo
    ang[:, i0-2:i0+3] = np.radians(grados)
    return dom, ang, flg


def kernel(dom, ang, flg, vx, vy, si, sj):
    vxo = np.full(nx*ny, -999.)
    vyo = np.full(nx*ny, -999.)
    CBO.cboveljih(ang.ravel(), flg.ravel(), dom.ravel(), vx.ravel().copy(), vy.ravel().copy(),
                  vxo, vyo, vp, vs, dx, dy, nx, ny, si, sj)
    return vxo.reshape(ny, nx), vyo.reshape(ny, nx)


def test_campo_uniforme_no_cambia():
    vx = np.full((ny, nx), 3.)
    vy = np.full((ny, nx), 4.)
    for tipo in range(1, 7):
        for g in (20., 45., 70.):
            dom, ang, flg = modelo(tipo, g)
            for si, sj in ((1, 1), (1, -1), (-1, -1)):
                ox, oy = kernel(dom, ang, flg, vx, vy, si, sj)
                if ox[20, 20] != -999.:                # el caso 5 con m < 1 no tiene fórmula
                    assert abs(ox[20, 20]-3) < 1e-9 and abs(oy[20, 20]-4) < 1e-9, (tipo, g, si, sj)


def test_reflejo_exacto():
    rng = np.random.default_rng(1)
    for tipo in range(1, 7):
        for g in (20., 35., 50., 70.):
            dom, ang, flg = modelo(tipo, g)
            vx = rng.standard_normal((ny, nx))
            vy = rng.standard_normal((ny, nx))
            a = kernel(dom, ang, flg, vx, vy, 1, 1)
            domr = np.zeros_like(dom)
            domr[:ny] = dom[:ny][::-1]
            b = kernel(domr, ang[::-1], flg[::-1], vx[::-1], -vy[::-1], 1, -1)
            m = a[0] != -999.
            assert np.allclose(a[0][m], b[0][::-1][m]) and np.allclose(a[1][m], -b[1][::-1][m])
