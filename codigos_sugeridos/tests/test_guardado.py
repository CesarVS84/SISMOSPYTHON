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


def test_topografia_desde_snif_npy(tmp_path):
    """Con snif.npy (SNIF_NPY) TOP usa esa superficie, la fuente [350, 293] y las estaciones del usuario."""
    import os, subprocess, sys
    import numpy as np
    n = 501
    fila = (330 + 40*np.clip(1-np.abs(np.arange(n)-259)/150, 0, 1)).astype(int)
    a = np.zeros((n, n))
    for i in range(n):
        a[fila[i], i] = 1
        a[fila[i]+1:, i] = 2
    # las estaciones E1-E7 deben quedar sobre la superficie: se fuerza su fila
    for e in ([170, 356], [190, 360], [210, 368], [261, 392], [281, 381], [301, 369], [321, 361]):
        a[:, e[0]] = 0
        a[e[1], e[0]] = 1
        a[e[1]+1:, e[0]] = 2
    np.save(tmp_path/'snif.npy', a)
    codigo = "import TOP; print(TOP.REAL, TOP.sour1, TOP.estaciones[7], int(TOP.js[170]))"
    env = dict(os.environ, SNIF_NPY=str(tmp_path/'snif.npy'))
    r = subprocess.run([sys.executable, '-c', codigo], capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split('\n')[-2] == 'True [350, %d] [200, 250] 356' % (int(np.argmax(a[:, 350] == 1)) - 60)
