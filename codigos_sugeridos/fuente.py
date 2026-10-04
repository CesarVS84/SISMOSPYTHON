# -*- coding: utf-8 -*-
"""
fuente.py: funciones temporales de la fuente.

Fuente por defecto (CF.TIPO_FUENTE = 'momento'): doble cupla con tasa de momento
gaussiana; ver más abajo (tasa_momento, tensor_unitario, sigma_gauss_para_fmax).

Fuente de velocidad impuesta (CF.TIPO_FUENTE = 'velocidad_pulso'), la del programa
original, con la misma forma:

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


def fuente_activa():
    """Función r(t) de la fuente de velocidad impuesta (solo con CF.TIPO_FUENTE = 'velocidad_pulso')."""
    return r


# ---------------------------------------------------------------------------
#  Fuente de momento (doble cupla): CF.TIPO_FUENTE = 'momento'
# ---------------------------------------------------------------------------
#
#  Una falla se representa con un tensor de momento M_ij(t) = M0 * m_ij * S(t), con S(t)
#  creciente de 0 a 1 (la falla se desliza) y m_ij el tensor unitario de la doble cupla.
#  En el esquema de velocidad-esfuerzo se inyecta como una tasa de esfuerzo en la celda de
#  la fuente:  dS_ij/dt += -dM_ij/dt / (dx*dy)   (ver CPML.inyectar_momento).
#
#  La tasa de momento es una suma de gaussianas (el evento principal y subeventos de la
#  ruptura). Una gaussiana tiene espectro gaussiano: sin lóbulos ni saltos, y la onda
#  radiada (velocidad ~ derivada de la tasa de momento) es bipolar, sin desplazamiento
#  permanente. Su ancho se calcula para que ESA onda radiada caiga NIVEL_FUENTE_DB a
#  FMAX_FUENTE (el límite de 8 Hz se refiere a lo que registran las estaciones).

def _nivel_velocidad(x):
    """|V(f)|/|V|max de la velocidad radiada de una gaussiana, con x = f/f_pico:
    V ~ f * exp(-(2 pi f sigma)^2 / 2), f_pico = 1/(2 pi sigma)."""
    return x*np.exp((1-x**2)/2)


def sigma_gauss_para_fmax(fmax=FMAX_FUENTE, nivel_db=NIVEL_FUENTE_DB):
    """Desviación estándar sigma (s) de la gaussiana de la tasa de momento tal que la
    velocidad radiada vale nivel_db (dB) respecto de su máximo a fmax."""
    objetivo = 10**(nivel_db/20.)
    lo, hi = 1., 50.                                # x > 1: rama decreciente
    for _ in range(200):
        medio = 0.5*(lo+hi)
        if _nivel_velocidad(medio) > objetivo:
            lo = medio
        else:
            hi = medio
    x = 0.5*(lo+hi)
    f_pico = fmax/x
    return 1./(2*np.pi*f_pico)


def _subeventos():
    """Tiempos (s) y pesos de la ruptura: evento principal en 0 y CF.SISMO_N subeventos con
    tiempo aleatorio en [0, SISMO_DURACION] (más densos al inicio) y amplitud exp(-t/SISMO_TAU).
    Los pesos suman 1 (el momento total es CF.M0). Semilla fija."""
    g = np.random.default_rng(CF.SISMO_SEMILLA)
    t = np.sort(g.uniform(0, 1, CF.SISMO_N)**1.5*CF.SISMO_DURACION)
    a = 0.6*g.uniform(0.3, 1.0, CF.SISMO_N)*np.exp(-t/CF.SISMO_TAU)
    t = np.concatenate([[0.], t])
    a = np.concatenate([[1.], a])
    return t, a/a.sum()


sigma_t = sigma_gauss_para_fmax()
t0 = 6*sigma_t                      # retraso para que la tasa de momento parta de ~0
_t_sub, _a_sub = _subeventos()


def tasa_momento(t):
    """dM0/dt (N/s) en el tiempo t: suma de gaussianas; integra a CF.M0."""
    t = np.asarray(t, dtype=float)
    x = (t[..., None] - t0 - _t_sub)/sigma_t
    return CF.M0*(_a_sub*np.exp(-0.5*x**2)).sum(axis=-1)/(sigma_t*np.sqrt(2*np.pi))


def tensor_unitario(angulo_grados=None):
    """(mxx, myy, mxy) de una doble cupla en el plano con el plano de falla a 'angulo' del eje x:
    M = ŝ n̂ᵀ + n̂ ŝᵀ, con ŝ = (cos φ, sin φ) y n̂ = (-sin φ, cos φ)."""
    phi = np.radians(CF.thetag if angulo_grados is None else angulo_grados)
    return -np.sin(2*phi), np.sin(2*phi), np.cos(2*phi)


def frecuencia_maxima(h, nivel_db=NIVEL_FUENTE_DB, fmax_busqueda=200.):
    """Frecuencia (Hz) donde el espectro de la fuente cae a nivel_db."""
    f = np.linspace(0, fmax_busqueda, 200001)
    esp = 20*np.log10(espectro_relativo(f, h))
    return f[np.argmax(esp <= nivel_db)]


if __name__ == '__main__':
    print('h1 = %.4f (original: 5)   tiempo del máximo: %.3f s' % (h1, np.log(_tasas(h1)[1]/_tasas(h1)[0])/(_tasas(h1)[1]-_tasas(h1)[0])))
    print('frecuencia a %.0f dB: %.2f Hz' % (NIVEL_FUENTE_DB, frecuencia_maxima(h1)))
    print('(con h = 5 del original:  %.2f Hz)' % frecuencia_maxima(5.))
