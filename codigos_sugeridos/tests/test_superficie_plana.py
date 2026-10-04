"""Superficie libre plana: la velocidad de la onda de Rayleigh debe estar a menos de
2 % del valor teórico (0.9194 vs para lambda = mu), y el modelo con masa efectiva
debe ser mejor que el aa = 1/rho del programa original."""
import numpy as np
import CF, TOP, CBA, CPML

nx, ny = 700, 300
dx = dy = 50.
dt = 0.004
nt = 1800
fc = 2.5


def ricker(t):
    t0 = 1.2/fc
    a = (np.pi*fc*(t-t0))**2
    return 5.*(1-2*a)*np.exp(-a)


def _lag(a, b):
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2*n))*np.fft.rfft(b, 2*n))[:n]
    k = int(np.argmax(cc))
    y0, y1, y2 = cc[k-1], cc[k], cc[k+1]
    return (k+0.5*(y0-y2)/(y0-2*y1+y2))*dt


def _velocidad(masa_efectiva, P=60, js0=240):
    m = TOP.modelo_desde_superficie(np.full(nx, float(js0)), nx=nx, ny=ny)
    aa = m['aa'] if masa_efectiva else np.where(m['rho'] > 0, 1/CF.Rho, 0.)
    ic = nx//2
    F = np.zeros((nx, ny))
    F[ic, js0-30:js0-3] = 50
    F[ic+1, js0-30:js0-3] = 90
    perf = CBA.crear_perfiles(nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, P1=P, P2=P, Q1=P, Q2=0, cp=CF.vp, f0=1.0)
    est = [[ic+150, js0], [ic+250, js0]]
    reg = CPML.simular(dict(aa=aa, lambdaa=m['lambdaa'], mu=m['mu'], F=F), perf, ricker, est, nx=nx, ny=ny, dx=dx, dy=dy,
                       dt=dt, nt=nt, damp=0., theta=np.radians(90.), P=(P, P, P, 0), verbose=False)
    return 100*dx/_lag(reg[0, :, 1], reg[1, :, 1])


def test_rayleigh():
    teorica = 0.9194*CF.vs
    c_efectiva = _velocidad(True)
    c_original = _velocidad(False)
    assert abs(c_efectiva/teorica-1) < 0.02
    assert abs(c_efectiva/teorica-1) < abs(c_original/teorica-1)
