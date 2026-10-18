# -*- coding: utf-8 -*-
"""
validacion_analitica.py: compara el solver (con la fuente de momento) con la solución analítica
de una doble cupla en un medio homogéneo e infinito (solucion_analitica.py).

Medio homogéneo de roca (sin superficie ni heterogeneidad ni atenuación), CPML alrededor, fuente
en el centro y ocho receptores en un círculo de radio R. Para cada receptor se compara la
velocidad (vx, vy) simulada con la analítica: error relativo L2 (tras alinear el retraso de
un paso a lo sumo), cociente de amplitudes y retraso.

Uso:   python validacion_analitica.py [--radio 3000] [--figura validacion.png]
"""

import argparse
import numpy as np

import CF
import CBA
import CPML
import TOP
import fuente
import solucion_analitica as SA


def simular_homogeneo(N=400, P=60, nt=1000, dt=CF.dt, dx=CF.dx, radio_celdas=60, sigma=CF.FUENTE_SIGMA):
    """Corre el solver en un medio homogéneo y devuelve (centro, receptores, registros)."""
    m = TOP.modelo_desde_superficie(np.full(N, N+10.), nx=N, ny=N)           # todo roca
    ic = N//2
    F = np.zeros((N, N))
    F[ic, ic] = 50
    F[ic+1, ic+1] = 90                                                      # punto medio: centro de la celda (ic, ic)
    perf = CBA.crear_perfiles(nx=N, ny=N, dx=dx, dy=dx, dt=dt, P1=P, P2=P, Q1=P, Q2=P, cp=CF.vp, f0=1.0)
    desplazamientos = []
    for ang in np.arange(8)*45.:
        a = np.radians(ang)
        desplazamientos.append((int(round(radio_celdas*np.cos(a))), int(round(radio_celdas*np.sin(a)))))
    receptores = [[ic+di, ic+dj] for di, dj in desplazamientos]
    reg = CPML.simular(dict(aa=m['aa'], lambdaa=m['lambdaa'], mu=m['mu'], F=F), perf, None, receptores,
                       nx=N, ny=N, dx=dx, dy=dx, dt=dt, nt=nt, damp=0., P=(P, P, P, P), verbose=False,
                       fuente_sigma=sigma, fuente_momento=(fuente.tasa_momento, fuente.tensor_unitario()))
    return ic, receptores, reg


def analitico(ic, receptores, nt, dt=CF.dt, dx=CF.dx, sigma=CF.FUENTE_SIGMA):
    mxx, myy, mxy = fuente.tensor_unitario()
    out = []
    for i, j in receptores:
        x, y = (i-(ic+0.5))*dx, (j-(ic+0.5))*dx                             # respecto del centro de la celda (ic, ic)
        t, vx, vy = SA.velocidad(x, y, fuente.tasa_momento, dt, CF.Rho, CF.vp, CF.vs, sigma*dx, mxx, myy, mxy)
        out.append(np.array([vx[:nt+1], vy[:nt+1]]).T)
    return np.array(out)


def comparar(**kw):
    nt = kw.get('nt', 1000)
    ic, rec, reg = simular_homogeneo(**kw)
    ana = analitico(ic, rec, nt, **{k: v for k, v in kw.items() if k in ('dt', 'dx', 'sigma')})
    filas = []
    for k in range(len(rec)):
        s, a = reg[k], ana[k]                                                # (nt+1, 2)
        mejor = None
        for desfase in (-1, 0, 1):
            if desfase >= 0:
                s2, a2 = s[desfase:], a[:len(a)-desfase]
            else:
                s2, a2 = s[:desfase], a[-desfase:]
            err = np.linalg.norm(s2-a2)/np.linalg.norm(a2)
            if mejor is None or err < mejor[0]:
                mejor = (err, desfase, np.vdot(a2, s2)/np.vdot(a2, a2))
        filas.append((k*45, mejor[0], mejor[2], mejor[1]))
    return ic, rec, reg, ana, np.array(filas)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--radio', type=int, default=60, help='radio de los receptores (celdas)')
    ap.add_argument('--figura', default=None)
    arg = ap.parse_args()
    ic, rec, reg, ana, f = comparar(radio_celdas=arg.radio)
    print('ángulo   error L2    amplitud sim/analítica   desfase (pasos)')
    for ang, e, esc, d in f:
        print('%5.0f°   %8.3f %%   %14.3f           %d' % (ang, 100*e, esc, d))
    print('error L2 medio: %.1f %%' % (100*f[:, 1].mean()))
    if arg.figura:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.arange(reg.shape[1])*CF.dt
        fig, ejes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
        for k, ax in enumerate(ejes.ravel()):
            ax.plot(t, np.hypot(*ana[k].T), color='#1f2933', lw=2.2, label='analítico |v|')
            ax.plot(t, np.hypot(*reg[k].T), color='#c2255c', lw=1.2, ls='--', label='solver |v|')
            ax.set_title('receptor a %d°  (error %.1f %%)' % (k*45, 100*f[k, 1]), fontsize=9)
        ejes[0, 0].legend(fontsize=8)
        fig.supxlabel('tiempo (s)'); fig.supylabel('velocidad (m/s)')
        fig.tight_layout(); fig.savefig(arg.figura, dpi=110)
        print('Guardado:', arg.figura)
