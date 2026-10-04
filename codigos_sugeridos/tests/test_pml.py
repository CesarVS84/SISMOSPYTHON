"""La CPML absorbe. Con la fuente del usuario repartida en una gaussiana (CF.FUENTE_SIGMA)
el eco queda bajo -50 dB respecto de la onda directa; con los dos nodos fijos del original
solo se llega a -23 dB (ondas cortas y obstáculo rígido que la capa no puede absorber).
La velocidad se promedia en 2x2 para quitar el modo de tablero."""
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


def _correr(N, P, sigma):
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
                 dt=dt, nt=nt, damp=0., theta=np.radians(30.), P=(P, P, P, P), verbose=False, registrar_campo=cb,
                 fuente_sigma=sigma)
    return np.array(sal)


def _error_db(sigma):
    ref = _correr(600, 0, sigma)             # dominio grande: sin ecos en la ventana
    rigido = _correr(200, 0, sigma)
    pml = _correr(200, 30, sigma)
    f = lambda x: 20*np.log10(np.abs(x-ref).max()/np.abs(ref).max())
    return f(rigido), f(pml)


def test_absorcion_fuente_suave():
    rigido, pml = _error_db(CF.FUENTE_SIGMA)
    assert pml < -50 and rigido - pml > 50


def test_fuente_de_dos_nodos_es_peor():
    _, pml_suave = _error_db(CF.FUENTE_SIGMA)
    _, pml_nodos = _error_db(None)
    assert pml_nodos > pml_suave + 20
