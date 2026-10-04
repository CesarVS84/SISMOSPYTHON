# -*- coding: utf-8 -*-
"""
animacion.py: animación de la onda sísmica (velocidades) sobre el modelo.

Une las instantáneas que guarda CPML.py (una cada 0.1 s) en un video MP4 (si hay
ffmpeg instalado) o en un GIF, con la topografía, el aire, la capa CPML, las
estaciones y el tiempo. La escala de colores es fija en toda la animación.

Uso (después de `python CPML.py`):
    python animacion.py                          # |v|, todas las instantáneas, salida/onda.mp4
    python animacion.py --campo vy --fps 12
    python animacion.py --paso 2 --salida onda.gif --dpi 70

Opciones:
    --campo      modulo (por defecto) | vx | vy
    --carpeta    carpeta con los resultados (por defecto 'salida')
    --salida     .mp4 o .gif (por defecto <carpeta>/onda.mp4, o .gif si no hay ffmpeg)
    --fps        cuadros por segundo (por defecto 10)
    --paso       usa una de cada N instantáneas (por defecto 1)
    --tmax       tiempo final (s); por defecto, hasta la última instantánea
    --dpi        resolución (por defecto 100)
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

import visual_comun as vc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--campo', choices=['modulo', 'vx', 'vy'], default='modulo')
    p.add_argument('--carpeta', default='salida')
    p.add_argument('--salida', default=None)
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--paso', type=int, default=1)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--escala', choices=['raiz', 'lineal'], default='raiz')
    a = p.parse_args()

    disp = vc.instantaneas_disponibles(a.carpeta)
    if a.tmax is not None:
        disp = [d for d in disp if d[0] <= a.tmax]
    disp = disp[::max(1, a.paso)]
    gamma = 0.5 if a.escala == 'raiz' else 1.0

    # escala fija: percentil del |campo| en unas pocas instantáneas repartidas en el tiempo
    muestra = [disp[k] for k in np.linspace(0, len(disp)-1, min(12, len(disp))).astype(int)]
    ejemplos = []
    for _, s in muestra:
        vx, vy = vc.cargar_campo(a.carpeta, s)
        ejemplos.append(vc.campo_derivado(vx, vy, a.campo)[0])
    vmax = vc.vmax_robusto(ejemplos, 99.9)
    _, con_signo, etiqueta = vc.campo_derivado(vx, vy, a.campo)

    fig, ax = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
    valores0 = vc.campo_derivado(*vc.cargar_campo(a.carpeta, disp[0][1]), a.campo)[0]
    im = vc.dibujar_mapa(ax, valores0, vmax, gamma, con_signo, titulo='', estaciones=True)
    vc.barra_color(fig, im, ax, vmax, gamma, con_signo, etiqueta)
    vc.leyenda_modelo(fig, y=-0.06)
    titulo = fig.suptitle('', fontsize=14, color=vc.TINTA, x=0.01, ha='left')

    def cuadro(k):
        t, s = disp[k]
        vx, vy = vc.cargar_campo(a.carpeta, s)
        valores = vc.campo_derivado(vx, vy, a.campo)[0]
        im.set_data(vc.escala_campo(valores, vmax, gamma, con_signo))
        titulo.set_text('Onda sísmica (%s)   t = %5.2f s' % (a.campo, t))
        return im, titulo

    salida = a.salida
    if salida is None:
        salida = os.path.join(a.carpeta, 'onda.mp4' if shutil.which('ffmpeg') else 'onda.gif')
    if salida.lower().endswith('.mp4') and not shutil.which('ffmpeg'):
        salida = salida[:-4] + '.gif'
        print('No hay ffmpeg: se guarda como GIF.')
    escritor = animation.FFMpegWriter(fps=a.fps, bitrate=2500) if salida.lower().endswith('.mp4') \
        else animation.PillowWriter(fps=a.fps)
    ani = animation.FuncAnimation(fig, cuadro, frames=len(disp), blit=False)
    ani.save(salida, writer=escritor, dpi=a.dpi, savefig_kwargs=dict(facecolor='white'))
    plt.close(fig)
    print('Guardado: %s  (%d cuadros, %.1f s de video)' % (salida, len(disp), len(disp)/a.fps))


if __name__ == '__main__':
    main()
