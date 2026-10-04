"""El núcleo fusionado (paso_velocidad/paso_esfuerzo) da lo mismo que el interior
más las funciones de la capa de CPMLD (cpml_vel/cpml_stress), verificadas contra el
código original."""
import numpy as np
from numba import njit, prange
import CF, TOP, CBA, CPMLD, CPML
from fuente import r as fuente_r

N, P, NT = 120, 20, 150
dx = dy = 50.
dt = 0.004


@njit(parallel=True, fastmath=True)
def _sb(Sbxx, Sbyy, Sbxy, Sxx, Syy, Sxy, nx, ny):
    for j in prange(1, ny):
        for i in range(1, nx):
            Sbxx[i+(nx+1)*j] = Sxx[i-1+(nx-1)*(j-1)]
            Sbyy[i+(nx+1)*j] = Syy[i-1+(nx-1)*(j-1)]
            Sbxy[i+(nx+1)*j] = Sxy[i-1+(nx-1)*(j-1)]


@njit(parallel=True, fastmath=True)
def _vel_interior(vo, vyo, vx, vy, Sbxx, Sbxy, Sbyy, aa, dt, dx, damp, nx, ny):
    for j in prange(ny):
        for i in range(nx):
            p = i+nx*j
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                vo[p] = 0.
                vyo[p] = 0.
            else:
                a1 = Sbxx[i+1+(nx+1)*j]-Sbxx[i+(nx+1)*j]+Sbxx[i+1+(nx+1)*(j+1)]-Sbxx[i+(nx+1)*(j+1)]
                a2 = Sbxy[i+(nx+1)*(j+1)]-Sbxy[i+(nx+1)*j]+Sbxy[i+1+(nx+1)*(j+1)]-Sbxy[i+1+(nx+1)*j]
                a3 = Sbxy[i+1+(nx+1)*j]-Sbxy[i+(nx+1)*j]+Sbxy[i+1+(nx+1)*(j+1)]-Sbxy[i+(nx+1)*(j+1)]
                a4 = Sbyy[i+(nx+1)*(j+1)]-Sbyy[i+(nx+1)*j]+Sbyy[i+1+(nx+1)*(j+1)]-Sbyy[i+1+(nx+1)*j]
                vo[p] = vx[p]*(1-damp)+0.5/dx*aa[p]*dt*(a1+a2)
                vyo[p] = vy[p]*(1-damp)+0.5/dx*aa[p]*dt*(a3+a4)


@njit(parallel=True, fastmath=True)
def _esf_interior(Sxxn, Syyn, Sxyn, Sxx, Syy, Sxy, vx, vy, b, c, d, dt, dx, nx, ny):
    for j in prange(ny-1):
        for i in range(nx-1):
            q = i+(nx-1)*j
            k = i+nx*j
            vxdx = vx[k+1]-vx[k]+vx[k+1+nx]-vx[k+nx]
            vydy = vy[k+nx]-vy[k]+vy[k+1+nx]-vy[k+1]
            vxdy = vx[k+nx]-vx[k]+vx[k+1+nx]-vx[k+1]
            vydx = vy[k+1]-vy[k]+vy[k+1+nx]-vy[k+nx]
            Sxxn[q] = Sxx[q]+0.5*dt/dx*(b[q]*vxdx+c[q]*vydy)
            Syyn[q] = Syy[q]+0.5*dt/dx*(b[q]*vydy+c[q]*vxdx)
            Sxyn[q] = Sxy[q]+0.5*dt/dx*d[q]*(vxdy+vydx)


def _referencia(m, perf, F, r, est, theta):
    nn, nc = N*N, (N-1)*(N-1)
    aa = m['aa']
    c = m['lambdaa']
    d = m['mu']
    b = c+2*d
    z = lambda n: np.zeros(n)
    vx, vy, vxn, vyn = z(nn), z(nn), z(nn), z(nn)
    S = {k: z(nc) for k in ('xx', 'yy', 'xy', 'xxn', 'yyn', 'xyn')}
    Sb = {k: z((N+1)*(N+1)) for k in ('xx', 'yy', 'xy')}
    mv, mvo = [z(nc) for _ in range(4)], [z(nc) for _ in range(4)]
    ms, mso = [z(nn) for _ in range(4)], [z(nn) for _ in range(4)]
    f50 = np.argwhere(F == 50)
    f90 = np.argwhere(F == 90)
    reg = np.zeros((len(est), NT+1, 2))
    idx = [i+N*j for i, j in est]
    for n in range(NT+1):
        rt = r(n*dt)
        for (i, j) in f50:
            vx[i+N*j] = rt*np.cos(theta); vy[i+N*j] = -rt*np.sin(theta)
        for (i, j) in f90:
            vx[i+N*j] = -rt*np.cos(theta); vy[i+N*j] = rt*np.sin(theta)
        _sb(Sb['xx'], Sb['yy'], Sb['xy'], S['xx'], S['yy'], S['xy'], N, N)
        _vel_interior(vxn, vyn, vx, vy, Sb['xx'], Sb['xy'], Sb['yy'], aa, dt, dx, CF.damp, N, N)
        # capa: usa los coeficientes SIN dividir por dx. El nodo del aire (aa = 0) no se mueve.
        CPMLD.cpml_vel(vxn, vyn, Sb['xx'], Sb['xy'], Sb['yy'], vx, vy, ms[0], ms[1], ms[2], ms[3], mso[0], mso[1], mso[2], mso[3],
                       perf['a_x'], perf['a_y'], perf['k_x'], perf['alpha_x'], perf['b_x'], perf['k_y'], perf['alpha_y'], perf['b_y'],
                       P, P, P, 0, aa, dx, dy, dt, CF.damp, N, N)
        reg[:, n, 0] = vxn[idx]
        reg[:, n, 1] = vyn[idx]
        _esf_interior(S['xxn'], S['yyn'], S['xyn'], S['xx'], S['yy'], S['xy'], vxn, vyn, b, c, d, dt, dx, N, N)
        CPMLD.cpml_stress(S['xxn'], S['xyn'], S['yyn'], S['xx'], S['xy'], S['yy'], vxn, vyn, mv[0], mv[1], mv[2], mv[3], mvo[0], mvo[1], mvo[2], mvo[3],
                          perf['a_x_half'], perf['a_y_half'], perf['k_x_half'], perf['alpha_x_half'], perf['b_x_half'], perf['k_y_half'], perf['alpha_y_half'], perf['b_y_half'],
                          P, P, P, 0, b, c, d, dx, dy, dt, N, N)
        vx, vxn = vxn, vx
        vy, vyn = vyn, vy
        for k in ('xx', 'yy', 'xy'):
            S[k], S[k+'n'] = S[k+'n'], S[k]
        mv, mvo = mvo, mv
        ms, mso = mso, ms
    return reg


def test_nucleo_fusionado_igual_a_la_referencia():
    hs = np.full(N, 80.) + 15*np.clip(1-np.abs(np.arange(N)-N//2)/30., 0, 1)       # volcán chico
    m = TOP.modelo_desde_superficie(hs, nx=N, ny=N)
    perf = CBA.crear_perfiles(nx=N, ny=N, dx=dx, dy=dy, dt=dt, P1=P, P2=P, Q1=P, Q2=0, cp=CF.vp)
    F = np.zeros((N, N))
    F[N//2, 60] = 50
    F[N//2+1, 61] = 90
    est = [[N//2-20, int(m['js'][N//2-20])], [N//2+20, int(m['js'][N//2+20])], [N//2, 50], [N//2+1, 61]]
    theta = np.radians(30.)
    ref = _referencia(m, perf, F, fuente_r, est, theta)
    modelo = dict(aa=m['aa'], lambdaa=m['lambdaa'], mu=m['mu'], F=F)
    reg = CPML.simular(modelo, perf, fuente_r, est, nx=N, ny=N, dx=dx, dy=dy, dt=dt, nt=NT, damp=CF.damp,
                       theta=theta, P=(P, P, P, 0), verbose=False)
    esc = np.abs(ref).max()
    assert esc > 1e-3
    assert np.abs(reg-ref).max()/esc < 1e-10
