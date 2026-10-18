# -*- coding: utf-8 -*-
"""
conversion_2d3d.py: de una traza 2-D a la amplitud 3-D equivalente.

El solver es 2-D: la fuente es una línea infinita y las ondas decaen como 1/sqrt(r), mientras
que un sismo real es una fuente puntual (1/r). Para campo lejano y un solo tipo de onda (P o
S) de velocidad c, a distancia r, la razón entre las funciones de Green 3-D y 2-D es

    G_3D / G_2D = exp(-i pi/4) sqrt(w / (2 pi c r))

(del comportamiento asintótico de H0^(1) y de exp(ikr)/(4 pi r)), es decir, una semiderivada en
el tiempo por 1/sqrt(2 pi c r). Si el solver se corrió con M0_2D = M0_3D/L (CF.M0), la traza 3-D
equivalente del mismo sismo de momento M0_3D es

    v_3D(t) = L * (2 pi c r)^(-1/2) * D^(1/2) v_2D(t)

con D^(1/2) la semiderivada (factor (i w)^(1/2) en la convención de NumPy).

Es una aproximación: vale lejos de la fuente (k r >> 1), para una sola fase y sin topografía ni
heterogeneidades que cambien la geometría del frente (ver Vidale y Helmberger, 1987). Verificada
en tests/test_conversion.py contra la fórmula de campo lejano 3-D de Aki y Richards (4.29).
"""

import numpy as np
import CF


def semiderivada(x, dt):
    """Semiderivada temporal de x (muestreada cada dt), con relleno de ceros para evitar el
    solapamiento circular."""
    n = len(x)
    N = int(2**np.ceil(np.log2(2*n)))
    w = 2*np.pi*np.fft.rfftfreq(N, dt)
    return np.fft.irfft(np.fft.rfft(x, N)*np.sqrt(w)*np.exp(1j*np.pi/4), N)[:n]


def traza_3d_equivalente(v2d, dt, r, c, longitud_falla=CF.LONGITUD_FALLA):
    """Traza 3-D equivalente (misma magnitud CF.MW) a la distancia r (m) de una traza 2-D del
    solver. c es la velocidad de la fase que se quiere (CF.vp o CF.vs). Ver la cabecera."""
    return longitud_falla*semiderivada(np.asarray(v2d, dtype=float), dt)/np.sqrt(2*np.pi*c*r)


def pgv_3d_equivalente(vx, vy, dt, r, c, longitud_falla=CF.LONGITUD_FALLA):
    """Velocidad máxima del suelo (módulo, m/s) 3-D equivalente de las trazas vx, vy."""
    ux = traza_3d_equivalente(vx, dt, r, c, longitud_falla)
    uy = traza_3d_equivalente(vy, dt, r, c, longitud_falla)
    return np.hypot(ux, uy).max()


if __name__ == '__main__':
    import sys
    import visual_comun as v
    carpeta = sys.argv[1] if len(sys.argv) > 1 else 'salida'
    nombres, VX, VY = v.cargar_estaciones(carpeta)
    dt = CF.nt*CF.dt/(VX.shape[1]-1)
    print('Mw = %.2f, M0_3D = %.2e N m, falla de %.0f m (M0_2D = %.2e N/m)' % (CF.MW, CF.M0_3D, CF.LONGITUD_FALLA, CF.M0))
    print('estación   r (km)   |v| máx 2-D (cm/s)   |v| máx 3-D equivalente (cm/s)   [fase S]')
    for k, n in enumerate(nombres):
        if n == 'F':
            continue
        r = v.distancia_a_la_fuente(n)
        print('%-8s %7.2f %18.2f %26.3f' % (n, r/1e3, 100*np.hypot(VX[k], VY[k]).max(),
                                             100*pgv_3d_equivalente(VX[k], VY[k], dt, r, CF.vs)))
