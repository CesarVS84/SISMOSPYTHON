# -*- coding: utf-8 -*-
"""
animacion.py: animaciones de la onda sísmica sobre el modelo.

A la izquierda, el campo de velocidad con la topografía, el aire, la capa CPML, las
estaciones y la fuente. A la derecha, dos sismogramas con un cursor en el tiempo actual:
el de la estación de la superficie más cercana a la fuente y el de la fuente (para ver el
pulso). Cada animación muestra una sola magnitud: vx, vy o el módulo |v|.

Usa las instantáneas que guarda CPML.py (una cada CF.SNAPSHOT_CADA = 0.01 s) y genera MP4
(si hay ffmpeg) o GIF. La escala de colores es fija en toda la animación.

Uso (después de `python CPML.py`):
    python animacion.py                      # tres videos: onda_vx, onda_vy y onda_modulo
    python animacion.py --campo vy
    python animacion.py --paso 10 --fps 15 --dpi 80      # más rápido y liviano

Opciones:
    --campo      todos (vx, vy y módulo, tres videos; por defecto) | vx | vy | modulo
    --carpeta    carpeta con los resultados (por defecto 'salida')
    --salida     prefijo de los archivos (por defecto <carpeta>/onda); termina en _<campo>.mp4
    --paso       usa una instantánea de cada N (por defecto 5: con 0.01 s por instantánea y
                 20 cuadros por segundo el video va en tiempo real)
    --fps        cuadros por segundo (por defecto 20)
    --tmax       tiempo final (s); por defecto, todo
    --dpi        resolución (por defecto 90)
    --formato    mp4 (por defecto, si hay ffmpeg) | gif
    --escala     raiz (realza las ondas débiles; por defecto) | lineal
"""

import argparse
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

import CF
import visual_comun as vc

ETIQUETA = {'vx': 'vx', 'vy': 'vy', 'modulo': '|v|'}


def _panel_sismograma(ax, t, y, nombre, componente, color):
    """Sismograma completo (tenue) y la parte ya transcurrida (intensa), con cursor."""
    ax.plot(t, y, color='#cbd2d9', lw=0.8, zorder=1)
    pasado, = ax.plot([], [], color=color, lw=1.2, zorder=2)
    punto, = ax.plot([], [], 'o', color=color, ms=5, mec='white', mew=1, zorder=4)
    cursor = ax.axvline(0, color=vc.TINTA, lw=0.9, zorder=3)
    ax.set_xlim(0, t[-1])
    m = np.abs(y).max()*1.08 or 1.0
    ax.set_ylim(0 if componente == 'modulo' else -m, m)
    ax.set_ylabel('%s (m/s)' % ETIQUETA[componente], color=vc.TINTA_SUAVE, fontsize=9)
    ax.set_title(nombre, loc='left', fontsize=10, color=vc.TINTA)
    ax.grid(color='#e4e7eb', lw=0.7)
    ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    for s in ax.spines.values():
        s.set_color('#9aa5b1')
    return pasado, punto, cursor


def crear_animacion(carpeta, campo, salida, paso, fps, tmax, dpi, gamma, formato):
    disp = vc.instantaneas_disponibles(carpeta)
    if tmax is not None:
        disp = [d for d in disp if d[0] <= tmax]
    disp = disp[::max(1, paso)]

    # estaciones: la de la superficie más cercana a la fuente y la fuente
    nombres, evx, evy = vc.cargar_estaciones(carpeta)
    cercana = vc.estacion_superficie_mas_cercana()
    te = np.arange(evx.shape[1])*CF.tfin/(evx.shape[1]-1)
    trazas = [(cercana, 'Estación %s (superficie, la más cercana a la fuente)' % cercana, '#0b7285'),
              ('F', 'Fuente (F): pulso impuesto', vc.FUENTE)]
    series = [(vc.serie(evx[nombres.index(n)], evy[nombres.index(n)], campo), tit, col) for n, tit, col in trazas]

    # escala de color fija
    muestra = [disp[k] for k in np.linspace(0, len(disp)-1, min(14, len(disp))).astype(int)]
    ejemplos = []
    for _, ident in muestra:
        vx, vy = vc.cargar_campo(carpeta, ident)
        ejemplos.append(vc.campo_derivado(vx, vy, campo)[0])
    vmax = vc.vmax_robusto(ejemplos, 99.9)
    _, con_signo, etiqueta = vc.campo_derivado(vx, vy, campo)

    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58], hspace=0.38, wspace=0.34, left=0.05, right=0.985, top=0.9, bottom=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    valores0 = vc.campo_derivado(*vc.cargar_campo(carpeta, disp[0][1]), campo)[0]
    im = vc.dibujar_mapa(ax, valores0, vmax, gamma, con_signo, titulo='', estaciones=True)
    vc.barra_color(fig, im, ax, vmax, gamma, con_signo, etiqueta)
    vc.leyenda_modelo(fig, y=0.005)
    paneles = [_panel_sismograma(a, te, s[0], s[1], campo, s[2]) for a, s in zip((ax1, ax2), series)]
    ax2.set_xlabel('tiempo (s)', color=vc.TINTA_SUAVE)
    plt.setp(ax1.get_xticklabels(), visible=False)
    titulo = fig.suptitle('', fontsize=14, color=vc.TINTA, x=0.01, ha='left')

    def cuadro(k):
        t, ident = disp[k]
        vx, vy = vc.cargar_campo(carpeta, ident)
        im.set_data(vc.escala_campo(vc.campo_derivado(vx, vy, campo)[0], vmax, gamma, con_signo))
        titulo.set_text('Onda sísmica: %s    t = %5.2f s' % (ETIQUETA[campo], t))
        n = int(np.searchsorted(te, t + 1e-9))
        for (pasado, punto, cursor), (y, _, _) in zip(paneles, series):
            pasado.set_data(te[:n], y[:n])
            punto.set_data([te[max(n-1, 0)]], [y[max(n-1, 0)]])
            cursor.set_xdata([t, t])
        return im, titulo

    if formato == 'mp4' and not shutil.which('ffmpeg'):
        formato = 'gif'
        print('No hay ffmpeg: se guarda como GIF.')
    archivo = '%s_%s.%s' % (salida, campo, formato)
    escritor = animation.FFMpegWriter(fps=fps, bitrate=3000) if formato == 'mp4' else animation.PillowWriter(fps=fps)
    ani = animation.FuncAnimation(fig, cuadro, frames=len(disp), blit=False)
    ani.save(archivo, writer=escritor, dpi=dpi, savefig_kwargs=dict(facecolor='white'))
    plt.close(fig)
    print('Guardado: %s  (%d cuadros, %.1f s de video)' % (archivo, len(disp), len(disp)/fps))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--campo', choices=['todos', 'vx', 'vy', 'modulo'], default='todos')
    p.add_argument('--carpeta', default='salida')
    p.add_argument('--salida', default=None)
    p.add_argument('--paso', type=int, default=5)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--dpi', type=int, default=90)
    p.add_argument('--formato', choices=['mp4', 'gif'], default='mp4')
    p.add_argument('--escala', choices=['raiz', 'lineal'], default='raiz')
    a = p.parse_args()
    salida = a.salida or os.path.join(a.carpeta, 'onda')
    for campo in (['vx', 'vy', 'modulo'] if a.campo == 'todos' else [a.campo]):
        crear_animacion(a.carpeta, campo, salida, a.paso, a.fps, a.tmax, a.dpi, 0.5 if a.escala == 'raiz' else 1.0, a.formato)


if __name__ == '__main__':
    main()
