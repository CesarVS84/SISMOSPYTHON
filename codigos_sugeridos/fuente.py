# -*- coding: utf-8 -*-
"""
fuente.py: función temporal de la fuente.

Misma forma que en el programa original:

    r(t) = w * ( exp(-(1-sqrt(z))/2 * H*t) - exp(-(1+sqrt(z))/2 * H*t) ),   H = 10*h

(respuesta de un oscilador sobreamortiguado: sube suavemente, tiene un máximo y
decae). La amplitud del máximo no depende de H; H solo cambia la escala de
tiempo. En el original h = 5 (H = 50), con lo que el espectro cae a -14 dB a 8 Hz y
a -40 dB recién cerca de 40 Hz. Aquí h se calcula para que el espectro haya caído a
NIVEL_FUENTE_DB (por defecto -20 dB, es decir 10 % de la amplitud máxima) a
FMAX_FUENTE (por defecto 8 Hz).

Espectro: |R(f)| = w*(a2-a1) / ( sqrt(a1^2+(2 pi f)^2) * sqrt(a2^2+(2 pi f)^2) ),
con a1 = (1-sqrt(z))/2*H y a2 = (1+sqrt(z))/2*H. Es máximo en f = 0 y decrece
de forma monótona, así que "frecuencia máxima" queda bien definida.
"""

import numpy as np
import CF
from CF import FMAX_FUENTE, NIVEL_FUENTE_DB

w = 1256.5
z = 0.0001


def _tasas(h):
    H = 10.*h
    return (1-np.sqrt(z))/2*H, (1+np.sqrt(z))/2*H


def espectro_relativo(f, h):
    """|R(f)| / |R(0)| de la fuente con parámetro h."""
    a1, a2 = _tasas(h)
    om = 2*np.pi*np.asarray(f, dtype=float)
    return (a1*a2)/np.sqrt((a1**2+om**2)*(a2**2+om**2))


def h_para_fmax(fmax=FMAX_FUENTE, nivel_db=NIVEL_FUENTE_DB):
    """Valor de h para el cual el espectro vale nivel_db (dB) en fmax."""
    objetivo = 10**(nivel_db/20.)
    lo, hi = 1e-6, 1e3                       # espectro_relativo(fmax, h) crece con h
    for _ in range(200):
        medio = np.sqrt(lo*hi)
        if espectro_relativo(fmax, medio) > objetivo:
            hi = medio
        else:
            lo = medio
    return np.sqrt(lo*hi)


#Parámetro h de la fuente de este programa (en el original: h1 = 5)
h1 = h_para_fmax()
#Las otras dos fuentes del original (s: más corta y de signo opuesto, o: más larga)
#conservan su proporción con h1 (h2 = 3*h1, h3 = h1/3)
h2 = 3*h1
h3 = h1/3


def _forma(t, h, signo=1.):
    a1, a2 = _tasas(h)
    return signo*w*(np.exp(-a1*t) - np.exp(-a2*t))


def r(t):
    return _forma(t, h1)


def s(t):
    return _forma(t, h2, -1.)


def o(t):
    return _forma(t, h3)


def _subeventos():
    """Tiempos y amplitudes (con signo) de la ruptura: evento principal en t = 0 y luego
    CF.SISMO_N subeventos con tiempos aleatorios en [0, SISMO_DURACION] (más densos al
    inicio), amplitud que decae como exp(-t/SISMO_TAU) y signo aleatorio. Se normaliza para
    que la energía espectral total sea la de un solo pulso: el contenido sobre 8 Hz es
    el mismo que el del pulso."""
    g = np.random.default_rng(CF.SISMO_SEMILLA)
    t = np.sort(g.uniform(0, 1, CF.SISMO_N)**1.5*CF.SISMO_DURACION)
    a = g.uniform(0.3, 1.0, CF.SISMO_N)*np.exp(-t/CF.SISMO_TAU)*g.choice([-1., 1.], CF.SISMO_N)
    t = np.concatenate([[0.], t])
    a = np.concatenate([[1.], 0.55*a])          # el evento principal domina
    return t, a/np.sqrt((a**2).sum())


_t_sub, _a_sub = _subeventos()


def sismo(t):
    """Ruptura de duración finita: suma de pulsos r(t - t_k) con amplitud a_k."""
    t = np.asarray(t, dtype=float)
    a1, a2 = _tasas(h1)
    tau = t[..., None] - _t_sub
    p = np.where(tau > 0, np.exp(-a1*np.maximum(tau, 0)) - np.exp(-a2*np.maximum(tau, 0)), 0.)
    return _escala*w*(p*_a_sub).sum(axis=-1)


def _calcular_escala():
    # mismo valor máximo que el pulso original (9.245), para que la amplitud sea comparable
    tt = np.arange(0, CF.SISMO_DURACION+3, 0.002)
    return 9.2448/np.abs(sismo(tt)).max()


_escala = 1.
_escala = _calcular_escala()


def fuente_activa():
    """Función de la fuente según CF.TIPO_FUENTE."""
    return sismo if CF.TIPO_FUENTE == 'sismo' else r


def frecuencia_maxima(h, nivel_db=NIVEL_FUENTE_DB, fmax_busqueda=200.):
    """Frecuencia (Hz) donde el espectro de la fuente cae a nivel_db."""
    f = np.linspace(0, fmax_busqueda, 200001)
    esp = 20*np.log10(espectro_relativo(f, h))
    return f[np.argmax(esp <= nivel_db)]


if __name__ == '__main__':
    print('h1 = %.4f (original: 5)   tiempo del máximo: %.3f s' % (h1, np.log(_tasas(h1)[1]/_tasas(h1)[0])/(_tasas(h1)[1]-_tasas(h1)[0])))
    print('frecuencia a %.0f dB: %.2f Hz' % (NIVEL_FUENTE_DB, frecuencia_maxima(h1)))
    print('(con h = 5 del original:  %.2f Hz)' % frecuencia_maxima(5.))
