# -*- coding: utf-8 -*-
"""
heterogeneidad.py: variaciones aleatorias suaves de la rigidez de la roca.

En un medio homogéneo solo la topografía dispersa las ondas, y el sismograma termina
pronto. En la roca real hay heterogeneidades a todas las escalas que producen la coda
(la cola de ondas dispersadas). Aquí se multiplica lambda y mu por (1 + e(x, y)), con e un
campo aleatorio gaussiano suave de desviación estándar CF.HETEROGENEIDAD (fracción) y
longitud de correlación CF.LONGITUD_CORRELACION (m). Se multiplican ambos por el mismo
factor, así que la razón vp/vs se conserva y las velocidades varían la mitad (en
fracción) que la rigidez. La densidad no cambia (el solver la usa en las masas nodales).
Con CF.HETEROGENEIDAD = 0 no se modifica nada.
"""

import numpy as np
import CF


def campo_aleatorio(forma, longitud, dx, semilla):
    """Campo gaussiano suave de media 0 y desviación estándar 1 (correlación gaussiana)."""
    g = np.random.default_rng(semilla)
    ruido = g.standard_normal(forma)
    ky = np.fft.fftfreq(forma[0], dx)[:, None]
    kx = np.fft.fftfreq(forma[1], dx)[None, :]
    filtro = np.exp(-0.5*(2*np.pi*longitud/np.sqrt(2))**2*(kx**2+ky**2))
    c = np.fft.ifft2(np.fft.fft2(ruido)*filtro).real
    return (c-c.mean())/c.std()


def perturbar(lambdaa, mu, nx=CF.nx, ny=CF.ny, dx=CF.dx):
    """(lambda, mu) perturbados; las celdas de aire (mu = 0) quedan igual."""
    if not CF.HETEROGENEIDAD:
        return lambdaa, mu
    # las celdas se guardan con índice i + (nx-1)*j: matriz (ny-1, nx-1)
    e = campo_aleatorio((ny-1, nx-1), CF.LONGITUD_CORRELACION, dx, CF.SEMILLA_HETEROGENEIDAD)
    fac = 1. + CF.HETEROGENEIDAD*np.clip(e, -3, 3).ravel()
    roca = np.asarray(mu) > 0
    return np.where(roca, lambdaa*fac, lambdaa), np.where(roca, mu*fac, mu)
