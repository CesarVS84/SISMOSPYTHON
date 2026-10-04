# -*- coding: utf-8 -*-
"""
espectrograma.py: espectrogramas de las estaciones (frecuencias sísmicas en el tiempo).

Para cada estación (E1..E9 y la fuente F) calcula la transformada de Fourier de
ventana corta (STFT, ventana de Hann) de la velocidad registrada y la dibuja como
mapa tiempo-frecuencia en dB. Marca la frecuencia máxima de la fuente (8 Hz).
Genera además una segunda figura con el espectro de amplitud de cada estación.

Uso (después de `python CPML.py`):
    python espectrograma.py
    python espectrograma.py --componente vy --ventana 3 --fmax 15
    python espectrograma.py --estaciones E1 E4 E8 F --salida espectro.png

Opciones:
    --componente    modulo (por defecto) | vx | vy
    --ventana       largo de la ventana en segundos (resolución en frecuencia = 1/ventana)
    --solape        fracción de solape entre ventanas (0 a <1; por defecto 0.9)
    --fmax          frecuencia máxima del eje (Hz); por defecto el doble de CF.FMAX_FUENTE
    --rango-db      rango dinámico de los colores en dB bajo el máximo (por defecto 50)
    --normalizar    estacion (cada una respecto de su máximo; por defecto) | global
    --estaciones    nombres de las estaciones a dibujar (E1 ... E9 F); por defecto todas
    --carpeta, --salida
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import CF
import visual_comun as vc


def stft(x, dt, ventana, solape):
    """STFT con ventana de Hann. Devuelve (t centro de cada ventana, f, |X|)."""
    nv = int(round(ventana/dt))
    salto = max(1, int(round(nv*(1-solape))))
    w = np.hanning(nv)
    x = x - x.mean()
    pos = np.arange(0, len(x)-nv+1, salto)
    nfft = 1 << int(np.ceil(np.log2(4*nv)))              # relleno con ceros: curvas más suaves
    X = np.array([np.abs(np.fft.rfft(w*x[p:p+nv], nfft)) for p in pos]).T
    f = np.fft.rfftfreq(nfft, dt)
    t = (pos + nv/2)*dt
    return t, f, X/ w.sum()*2


def serie(vx, vy, componente):
    if componente == 'vx':
        return vx
    if componente == 'vy':
        return vy
    return np.hypot(vx, vy)


def figura_espectrogramas(nombres, vx, vy, componente, ventana, solape, fmax, rango_db, normalizar):
    dt = CF.tfin/(vx.shape[1]-1)
    sig = [serie(vx[k], vy[k], componente) for k in range(len(nombres))]
    res = [stft(s, dt, ventana, solape) for s in sig]
    ref_global = max(r[2].max() for r in res)
    n = len(nombres)
    nc = min(n, 3)
    nf = int(np.ceil(n/nc))
    fig, axes = plt.subplots(nf, nc, figsize=(5.4*nc+1.2, 3.5*nf+1.0), squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    pcm = None
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis('off')
            continue
        t, f, X = res[k]
        ref = ref_global if normalizar == 'global' else X.max()
        db = 20*np.log10(np.maximum(X, 1e-30)/ref)
        sel = f <= fmax
        pcm = ax.pcolormesh(t, f[sel], db[sel], cmap='viridis', vmin=-rango_db, vmax=0, shading='auto', rasterized=True)
        ax.axhline(CF.FMAX_FUENTE, color='white', lw=1.2, ls=(0, (5, 3)))
        ax.axhline(CF.FMAX_FUENTE, color=vc.TINTA, lw=0.6, ls=(0, (5, 3)))
        ax.set_title(nombres[k] if nombres[k] != 'F' else 'F (fuente)', loc='left', fontsize=11, color=vc.TINTA)
        ax.set_ylim(0, fmax)
        ax.set_xlim(0, CF.tfin)
        ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
        for s in ax.spines.values():
            s.set_color('#9aa5b1')
    for ax in axes[-1]:
        ax.set_xlabel('tiempo (s)', color=vc.TINTA_SUAVE)
    for ax in axes[:, 0]:
        ax.set_ylabel('frecuencia (Hz)', color=vc.TINTA_SUAVE)
    axes[0, 0].annotate('%.0f Hz (fuente)' % CF.FMAX_FUENTE, (CF.tfin*0.98, CF.FMAX_FUENTE), xytext=(0, 4),
                        textcoords='offset points', ha='right', fontsize=8, color='white',
                        path_effects=[__import__('matplotlib.patheffects', fromlist=['x']).withStroke(linewidth=2, foreground=vc.TINTA)])
    cb = fig.colorbar(pcm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label('amplitud (dB respecto del máximo %s)' % ('global' if normalizar == 'global' else 'de cada estación'),
                 fontsize=9, color=vc.TINTA_SUAVE)
    cb.ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    fig.suptitle('Espectrogramas de las estaciones (%s) · ventana %.1f s' % (componente, ventana), fontsize=13,
                 color=vc.TINTA, x=0.01, ha='left')
    return fig


def figura_espectros(nombres, vx, vy, componente, fmax):
    """Espectro de amplitud (Fourier de toda la señal) de cada estación, en dB."""
    dt = CF.tfin/(vx.shape[1]-1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    colores = plt.get_cmap('tab10')
    mx = 0
    espectros = []
    for k in range(len(nombres)):
        s = serie(vx[k], vy[k], componente)
        s = (s - s.mean())*np.hanning(len(s))
        X = np.abs(np.fft.rfft(s, 1 << int(np.ceil(np.log2(len(s)*4)))))
        f = np.fft.rfftfreq(1 << int(np.ceil(np.log2(len(s)*4))), dt)
        espectros.append((f, X))
        mx = max(mx, X.max())
    for k, (f, X) in enumerate(espectros):
        sel = f <= fmax
        ax.plot(f[sel], 20*np.log10(np.maximum(X[sel], 1e-30)/mx), lw=1.4, color=colores(k % 10), label=nombres[k])
    ax.axvline(CF.FMAX_FUENTE, color=vc.TINTA, lw=1, ls=(0, (5, 3)))
    ax.text(CF.FMAX_FUENTE, 2, ' %.0f Hz' % CF.FMAX_FUENTE, color=vc.TINTA_SUAVE, fontsize=9, va='bottom')
    ax.set_xlim(0, fmax)
    ax.set_ylim(-90, 5)
    ax.set_xlabel('frecuencia (Hz)', color=vc.TINTA_SUAVE)
    ax.set_ylabel('amplitud (dB respecto del máximo)', color=vc.TINTA_SUAVE)
    ax.set_title('Espectro de amplitud de las estaciones (%s)' % componente, loc='left', color=vc.TINTA)
    ax.grid(color='#e4e7eb', lw=0.8)
    ax.tick_params(colors=vc.TINTA_SUAVE, labelsize=9)
    ax.legend(ncol=5, frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    for s in ax.spines.values():
        s.set_color('#9aa5b1')
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--carpeta', default='salida')
    p.add_argument('--salida', default=None)
    p.add_argument('--componente', choices=['modulo', 'vx', 'vy'], default='modulo')
    p.add_argument('--ventana', type=float, default=2.0)
    p.add_argument('--solape', type=float, default=0.9)
    p.add_argument('--fmax', type=float, default=2*CF.FMAX_FUENTE)
    p.add_argument('--rango-db', type=float, default=50.)
    p.add_argument('--normalizar', choices=['estacion', 'global'], default='estacion')
    p.add_argument('--estaciones', nargs='+', default=None)
    a = p.parse_args()

    nombres, vx, vy = vc.cargar_estaciones(a.carpeta)
    if a.estaciones:
        idx = [nombres.index(e) for e in a.estaciones if e in nombres]
        nombres = [nombres[i] for i in idx]
        vx, vy = vx[idx], vy[idx]
    salida = a.salida or os.path.join(a.carpeta, 'espectrograma.png')
    base, ext = os.path.splitext(salida)
    fig = figura_espectrogramas(nombres, vx, vy, a.componente, a.ventana, a.solape, a.fmax, a.rango_db, a.normalizar)
    fig.savefig(salida, dpi=130, bbox_inches='tight', facecolor='white')
    print('Guardado:', salida)
    fig2 = figura_espectros(nombres, vx, vy, a.componente, a.fmax)
    fig2.savefig(base + '_espectros' + (ext or '.png'), dpi=130, bbox_inches='tight', facecolor='white')
    print('Guardado:', base + '_espectros' + (ext or '.png'))


if __name__ == '__main__':
    main()
