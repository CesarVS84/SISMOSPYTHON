"""La CPML absorbe: con la fuente del usuario (par de nodos en diagonal) y la
velocidad promediada en 2x2 (se elimina el modo de tablero, que ni se propaga ni se
absorbe), el eco es al menos 20 dB menor que con una pared rígida."""
import numpy as np
import CF, TOP, CBA, CPML

dx = dy = 50.
dt = 0.004
nt = 1000
fc = 3.


def ricker(t):
    t0 = 1.5/fc
    a = (np.pi*fc*(t-t0))**2
    return 5.*(1-2*a)*np.exp(-a)


def _correr(N, P):
    m = TOP.modelo_desde_superficie(np.full(N, N+10.), nx=N, ny=N)           # todo roca
    ic = N//2
    F = np.zeros((N, N))
    F[ic, ic] = 50
    F[ic+1, ic+1] = 90
    perf = CBA.crear_perfiles(nx=N, ny=N, dx=dx, dy=dy, dt=dt, P1=P, P2=P, Q1=P, Q2=P, cp=CF.vp, f0=1.0)
    k0 = ic+60+N*ic
    sal = []

    def cb(n, vx, vy):
        sal.append(0.25*(vx[k0]+vx[k0+1]+vx[k0+N]+vx[k0+N+1]))
    CPML.simular(dict(aa=m['aa'], lambdaa=m['lambdaa'], mu=m['mu'], F=F), perf, ricker, [[ic, ic]], nx=N, ny=N, dx=dx, dy=dy,
                 dt=dt, nt=nt, damp=0., theta=np.radians(30.), P=(P, P, P, P), verbose=False, registrar_campo=cb)
    return np.array(sal)


def test_absorcion():
    ref = _correr(600, 0)                    # dominio grande: sin ecos en la ventana
    rigido = _correr(200, 0)
    pml = _correr(200, 30)
    e_rig = np.abs(rigido-ref).max()/np.abs(ref).max()
    e_pml = np.abs(pml-ref).max()/np.abs(ref).max()
    assert 20*np.log10(e_rig/e_pml) > 20
