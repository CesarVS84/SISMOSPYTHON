# -*- coding: utf-8 -*-
"""
solucion_analitica.py: campo de una doble cupla en un medio homogéneo e infinito (2-D, plano
P-SV), para validar el solver y la fuente de momento.

Convención de tiempo exp(-i w t) (la transformada de NumPy usa la opuesta: en `velocidad` se conjuga G). La función de Green del desplazamiento i por una fuerza
unitaria en k en el origen es

    G_ik = 1/rho * [ delta_ik g_s / c_s^2 + 1/w^2 d_i d_k (g_s - g_p) ],   g_c = (i/4) H0^(1)(w r / c_c)

(verificada: cumple (mu lap + rho w^2) u + (lambda + mu) grad div u = -F delta fuera del origen,
y la divergencia es la de una onda P pura; ver tests/test_analitica.py). Una fuente de momento
M_jk(t) equivale a la fuerza f_k = -d_j (M_jk delta), de modo que

    u_i = - M_jk d_j G_ik        y, con M(t) = M0 m_jk S(t),   v_i(w) = - m_jk d_j G_ik * Mdot(w),

con Mdot = dM/dt (fuente.tasa_momento). El solver reparte la fuente en una gaussiana de
desviación sigma (en m): cada onda de número k queda multiplicada por exp(-k^2 sigma^2/2).
"""

import numpy as np
from scipy.special import hankel1


def _g(x, y, w, c, sigma):
    """g_c suavizada: (i/4) H0^(1)(k r) exp(-k^2 sigma^2/2), k = w/c."""
    k = w/c
    r = np.hypot(x, y)
    return 0.25j*hankel1(0, k*r)*np.exp(-0.5*(k*sigma)**2)


def _d(f, eje, h):
    """Derivada central respecto de x (eje 0) o y (eje 1) de f(x, y)."""
    if eje == 0:
        return lambda x, y: (f(x+h, y) - f(x-h, y))/(2*h)
    return lambda x, y: (f(x, y+h) - f(x, y-h))/(2*h)


def derivada_G(x, y, w, rho, cp, cs, sigma, i, k, j, h=3.):
    """d_j G_ik(x, y; w) (arreglo en w). Derivadas centrales anidadas de g (paso h, m)."""
    gs = lambda xx, yy: _g(xx, yy, w, cs, sigma)
    gp = lambda xx, yy: _g(xx, yy, w, cp, sigma)
    ejes = (i, k, j)

    def deriv(g):
        f = g
        for e in ejes:
            f = _d(f, e, h)
        return f(x, y)
    termino = deriv(gs) - deriv(gp)
    # delta_ik g_s/c_s^2 derivado respecto de j
    if i == k:
        termino = termino/w**2 + _d(gs, j, h)(x, y)/cs**2
    else:
        termino = termino/w**2
    return termino/rho


def velocidad(x, y, tasa, dt, rho, cp, cs, sigma, mxx, myy, mxy, T=60., retraso=0.):
    """(vx, vy)(t) en (x, y) m respecto de la fuente, para la tasa de momento tasa(t) (N/s),
    en los tiempos n*dt (con retraso: la señal se retrasa esa cantidad de segundos). Se rellena con ceros hasta T segundos para evitar el solapamiento."""
    n = int(round(T/dt))
    t = np.arange(n)*dt
    Md = np.fft.rfft(tasa(t))
    w = 2*np.pi*np.fft.rfftfreq(n, dt)
    w[0] = w[1]/50.                         # sin singularidad en f = 0 (la velocidad ahí es ~0)
    m = {(0, 0): mxx, (1, 1): myy, (0, 1): mxy, (1, 0): mxy}
    v = []
    for i in (0, 1):
        s = 0.
        for (j, k), mjk in m.items():
            s = s - mjk*derivada_G(x, y, w, rho, cp, cs, sigma, i, k, j)
        # NumPy transforma con exp(-i w t) y G está dada para exp(+... ver cabecera): se usa su conjugado
        v.append(np.fft.irfft(np.conj(s)*Md*np.exp(-1j*w*retraso), n))
    return t, v[0], v[1]
