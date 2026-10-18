"""Validación contra la solución analítica de una doble cupla en un medio homogéneo."""
import numpy as np
import CF
import fuente
import solucion_analitica as SA
import validacion_analitica as V

rho, cp, cs = CF.Rho, CF.vp, CF.vs


def test_green_cumple_la_ecuacion_de_onda():
    lam, mu = rho*(cp**2-2*cs**2), rho*cs**2
    w = np.array([2*np.pi*3.])
    x0, y0, h, h2 = 2500., 1500., 3., 20.
    gs = lambda x, y: SA._g(x, y, w, cs, 0.)
    gp = lambda x, y: SA._g(x, y, w, cp, 0.)

    def U(x, y, i, con_delta=True):
        f = lambda g: SA._d(SA._d(g, i, h), 0, h)(x, y)
        t = (f(gs)-f(gp))/w**2
        if i == 0 and con_delta:
            t = t + gs(x, y)/cs**2
        return t/rho

    def residuo(con_delta):
        ux = lambda x, y: U(x, y, 0, con_delta)
        uy = lambda x, y: U(x, y, 1, con_delta)
        dxx = lambda f: (f(x0+h2, y0)-2*f(x0, y0)+f(x0-h2, y0))/h2**2
        dyy = lambda f: (f(x0, y0+h2)-2*f(x0, y0)+f(x0, y0-h2))/h2**2
        dxy = lambda f: (f(x0+h2, y0+h2)-f(x0+h2, y0-h2)-f(x0-h2, y0+h2)+f(x0-h2, y0-h2))/(4*h2**2)
        rx = mu*(dxx(ux)+dyy(ux)) + rho*w**2*ux(x0, y0) + (lam+mu)*(dxx(ux)+dxy(uy))
        ry = mu*(dxx(uy)+dyy(uy)) + rho*w**2*uy(x0, y0) + (lam+mu)*(dxy(ux)+dyy(uy))
        esc = rho*w**2*np.hypot(abs(ux(x0, y0)), abs(uy(x0, y0)))
        return np.hypot(abs(rx), abs(ry)).item()/esc.item()
    assert residuo(True) < 5e-3
    assert residuo(False) > 0.5                  # la prueba distingue una G equivocada


def test_solucion_analitica_es_causal():
    mxx, myy, mxy = fuente.tensor_unitario()
    dt = 0.005
    x, y = 3000., 0.
    t, vx, vy = SA.velocidad(x, y, fuente.tasa_momento, dt, rho, cp, cs, 75., mxx, myy, mxy)
    llegada = x/cp
    antes = t < llegada-0.15
    assert np.abs(vx[antes]).max() < 0.02*np.abs(vx).max()


def test_solver_coincide_con_la_analitica():
    ic, rec, reg, ana, f = V.comparar(N=300, P=50, nt=800, radio_celdas=60)
    assert np.all(np.abs(f[:, 2]-1) < 0.03)      # amplitud absoluta, signo y orientación
    assert f[:, 1].max() < 0.25                  # forma de onda (queda dispersión numérica de la malla)
