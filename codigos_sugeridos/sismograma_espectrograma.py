# -*- coding: utf-8 -*-
"""
sismograma_espectrograma.py: figuras con el sismograma y el espectrograma de cada estación.

Para cada estación genera una figura con tres paneles que comparten el tiempo: el
sismograma de vx, el de vy y el espectrograma (en dB) de la componente elegida. Genera
además una figura resumen con todas las estaciones (sismograma del módulo |v| a la
izquierda y espectrograma a la derecha).

Uso (después de `python CPML.py`):
    python sismograma_espectrograma.py
    python sismograma_espectrograma.py --estaciones E3 F --componente vy
    python sismograma_espectrograma.py --ventana 3 --fmax 12

Opciones:
    --carpeta       carpeta con los resultados (por defecto 'salida')
    --salida        carpeta de las figuras (por defecto <carpeta>/figuras)
    --estaciones    estaciones a dibujar (E1 ... E9 F); por defecto todas
    --componente    modulo (por defecto) | vx | vy: componente del espectrograma
    --ventana       largo de la ventana de la STFT en segundos (por defecto 2)
    --fmax          frecuencia máxima del eje (Hz); por defecto el doble de CF.FMAX_FUENTE
    --rango-db      rango dinámico del espectrograma en dB (por defecto 50)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import CF
import visual_comun as vc


def _estilo(ax):
    ax.grid(color='#e4e7eb', lw=0.7)
    ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    for s in ax.spines.values():
        s.set_color('#9aa5b1')


def _sismograma(ax, t, y, etiqueta, color, con_signo=True):
    ax.plot(t, y, color=color, lw=0.9)
    m = np.abs(y).max()*1.08 or 1.
    ax.set_ylim(-m if con_signo else 0, m)
    ax.set_xlim(0, t[-1])
    ax.set_ylabel(etiqueta, color=vc.TINTA_SUAVE, fontsize=9)
    _estilo(ax)


def _espectrograma(ax, y, dt, ventana, fmax, rango_db):
    t, f, db = vc.espectrograma_db(y, dt, ventana, 0.9, fmax, rango_db)
    pcm = ax.pcolormesh(t, f, db, cmap='viridis', vmin=-rango_db, vmax=0, shading='auto', rasterized=True)
    ax.axhline(CF.FMAX_FUENTE, color='white', lw=1.2, ls=(0, (5, 3)))
    ax.axhline(CF.FMAX_FUENTE, color=vc.TINTA, lw=0.6, ls=(0, (5, 3)))
    ax.set_ylim(0, fmax)
    ax.set_ylabel('frecuencia (Hz)', color=vc.TINTA_SUAVE, fontsize=9)
    ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    for s in ax.spines.values():
        s.set_color('#9aa5b1')
    return pcm


def figura_estacion(nombre, vx, vy, componente, ventana, fmax, rango_db):
    dt = CF.tfin/(len(vx)-1)
    t = np.arange(len(vx))*dt
    pos = vc.posiciones_estaciones()[nombre]
    fi, fj = vc.TOP.sour1
    dist = np.hypot(pos[0]-fi, pos[1]-fj)*CF.dx/1000
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.7]),
                             constrained_layout=True)
    _sismograma(axes[0], t, vx, 'vx (m/s)', '#0b7285')
    _sismograma(axes[1], t, vy, 'vy (m/s)', '#5f3dc4')
    pcm = _espectrograma(axes[2], vc.serie(vx, vy, componente), dt, ventana, fmax, rango_db)
    axes[2].set_xlabel('tiempo (s)', color=vc.TINTA_SUAVE)
    cb = fig.colorbar(pcm, ax=axes[2], pad=0.01, fraction=0.04)
    cb.set_label('amplitud (dB respecto del máximo)', fontsize=8, color=vc.TINTA_SUAVE)
    cb.ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    titulo = 'Fuente (F)' if nombre == 'F' else 'Estación %s' % nombre
    fig.suptitle('%s · sismograma y espectrograma (%s) · %s' % (
        titulo, componente, 'nodo (%d, %d)' % pos if nombre == 'F' else 'nodo (%d, %d), a %.1f km de la fuente' % (pos[0], pos[1], dist)),
        fontsize=12, color=vc.TINTA, x=0.01, ha='left')
    return fig


def figura_resumen(nombres, vx, vy, ventana, fmax, rango_db):
    n = len(nombres)
    dt = CF.tfin/(vx.shape[1]-1)
    t = np.arange(vx.shape[1])*dt
    fig, axes = plt.subplots(n, 2, figsize=(14, 1.9*n+1.2), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    for k, nom in enumerate(nombres):
        mod = np.hypot(vx[k], vy[k])
        _sismograma(axes[k, 0], t, mod, '%s  |v|' % nom, '#0b7285', con_signo=False)
        pcm = _espectrograma(axes[k, 1], mod, dt, ventana, fmax, rango_db)
        axes[k, 1].set_ylabel('Hz', color=vc.TINTA_SUAVE, fontsize=8)
    axes[0, 0].set_title('Sismograma (|v|, m/s)', loc='left', fontsize=10, color=vc.TINTA)
    axes[0, 1].set_title('Espectrograma (dB respecto del máximo de cada estación)', loc='left', fontsize=10, color=vc.TINTA)
    for a in axes[-1]:
        a.set_xlabel('tiempo (s)', color=vc.TINTA_SUAVE)
    cb = fig.colorbar(pcm, ax=axes[:, 1], fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--carpeta', default='salida')
    p.add_argument('--salida', default=None)
    p.add_argument('--estaciones', nargs='+', default=None)
    p.add_argument('--componente', choices=['modulo', 'vx', 'vy'], default='modulo')
    p.add_argument('--ventana', type=float, default=2.0)
    p.add_argument('--fmax', type=float, default=2*CF.FMAX_FUENTE)
    p.add_argument('--rango-db', type=float, default=50.)
    a = p.parse_args()
    nombres, vx, vy = vc.cargar_estaciones(a.carpeta)
    salida = a.salida or os.path.join(a.carpeta, 'figuras')
    os.makedirs(salida, exist_ok=True)
    sel = a.estaciones or nombres
    for nom in sel:
        k = nombres.index(nom)
        fig = figura_estacion(nom, vx[k], vy[k], a.componente, a.ventana, a.fmax, a.rango_db)
        archivo = os.path.join(salida, 'estacion_%s.png' % nom)
        fig.savefig(archivo, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('Guardado:', archivo)
    idx = [nombres.index(n) for n in sel]
    fig = figura_resumen([nombres[i] for i in idx], vx[idx], vy[idx], a.ventana, a.fmax, a.rango_db)
    archivo = os.path.join(salida, 'resumen_estaciones.png')
    fig.savefig(archivo, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Guardado:', archivo)


if __name__ == '__main__':
    main()
