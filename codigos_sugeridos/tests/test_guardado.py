"""El guardado de campos completos cada 0.01 s."""
import numpy as np
import CF, TOP, CBA, CPML
from fuente import r


def test_guardado_cada_centesima(tmp_path):
    N, dt, nt = 60, 0.005, 40
    m = TOP.modelo_desde_superficie(np.full(N, N+10.), nx=N, ny=N)
    F = np.zeros((N, N))
    F[30, 30] = 50
    F[31, 31] = 90
    perf = CBA.crear_perfiles(nx=N, ny=N, dx=50., dy=50., dt=dt, P1=10, P2=10, Q1=10, Q2=10, cp=CF.vp)
    reg = CPML.simular(dict(aa=m['aa'], lambdaa=m['lambdaa'], mu=m['mu'], F=F), perf, r, [[20, 30]], nx=N, ny=N,
                       dx=50., dy=50., dt=dt, nt=nt, P=(10, 10, 10, 10), verbose=False, carpeta=str(tmp_path),
                       guardar_cada=0.01, fuente_sigma=1.5, instantaneas=None)
    vx = np.load(tmp_path/'campo_vx.npy', mmap_mode='r')
    t = np.load(tmp_path/'tiempos.npy')
    assert vx.shape == (nt//2+1, N, N) and len(t) == nt//2+1
    assert abs(t[1]-t[0]-0.01) < 1e-12
    # la instantánea del paso 20 coincide con el registro de la estación (i=20, j=30)
    k = 20//2
    assert abs(vx[k, 30, 20] - reg[0, 20, 0]) < 1e-6*max(1., abs(reg[0, 20, 0])) + 1e-9
    assert np.abs(vx[-1]).max() > 0
