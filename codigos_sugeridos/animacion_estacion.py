# -*- coding: utf-8 -*-
"""
animacion_estacion.py: una animación por estación con su sismograma y su espectrograma.

Cada video marca la llegada teórica de las ondas P (naranja) y S (magenta) y cambia de color
la estación en el esquema al llegar cada una. Muestra cómo se va construyendo, con el tiempo, el sismograma (vx y vy) y el
espectrograma de la estación, con un cursor en el tiempo actual. A la derecha, un
esquema del modelo con la posición de la estación. Se genera un video por estación
(E1 ... E9) y uno para la fuente (F).

Uso (después de `python CPML.py`):
    python animacion_estacion.py
    python animacion_estacion.py --estaciones E3 F
    python animacion_estacion.py --paso 0.2 --fps 12 --dpi 70      # más rápido

Opciones:
    --carpeta       carpeta con los resultados (por defecto 'salida')
    --salida        carpeta de los videos (por defecto <carpeta>/animaciones_estaciones)
    --estaciones    estaciones a animar; por defecto todas
    --componente    modulo (por defecto) | vx | vy: componente del espectrograma
    --paso          segundos de simulación entre cuadros (por defecto 0.1)
    --fps           cuadros por segundo (por defecto 15)
    --ventana       largo de la ventana de la STFT (s; por defecto 2)
    --fmax          frecuencia máxima del eje (Hz); por defecto el doble de CF.FMAX_FUENTE
    --rango-db      rango dinámico del espectrograma (dB; por defecto 50)
    --dpi           resolución (por defecto 90)
    --formato       mp4 (por defecto, si hay ffmpeg) | gif
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


def _estilo(ax):
    ax.grid(color='#e4e7eb', lw=0.7)
    ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    for s in ax.spines.values():
        s.set_color('#9aa5b1')


def animar_estacion(nombre, vx, vy, carpeta_salida, componente, paso, fps, ventana, fmax, rango_db, dpi, formato):
    dt = CF.tfin/(len(vx)-1)
    t = np.arange(len(vx))*dt
    pos = vc.posiciones_estaciones()[nombre]
    ts, fs, db = vc.espectrograma_db(vc.serie(vx, vy, componente), dt, ventana, 0.9, fmax, rango_db)
    cuadros = np.arange(0, CF.tfin + 1e-9, paso)

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.34], height_ratios=[1, 1, 1.6], hspace=0.28, wspace=0.12,
                          left=0.07, right=0.985, top=0.9, bottom=0.08)
    axx, axy, axs = (fig.add_subplot(gs[k, 0]) for k in range(3))
    axm = fig.add_subplot(gs[:, 1])

    t_p, t_s = vc.tiempos_llegada(nombre)
    lineas, cursores = [], []
    for ax, y, etiqueta, color in ((axx, vx, 'vx (m/s)', '#0b7285'), (axy, vy, 'vy (m/s)', '#5f3dc4')):
        ax.plot(t, y, color='#cbd2d9', lw=0.8)
        ln, = ax.plot([], [], color=color, lw=1.2)
        cur = ax.axvline(0, color=vc.TINTA, lw=0.9)
        m = np.abs(y).max()*1.08 or 1.
        ax.set_ylim(-m, m)
        ax.set_xlim(0, CF.tfin)
        ax.set_ylabel(etiqueta, color=vc.TINTA_SUAVE, fontsize=9)
        _estilo(ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        if nombre != 'F':
            for tt, lab, col in ((t_p, 'P', vc.COLOR_P), (t_s, 'S', vc.COLOR_S)):
                ax.axvline(tt, color=col, lw=1.4, ls=(0, (4, 2)))
                ax.annotate(lab, (tt, 1.0), xycoords=('data', 'axes fraction'), xytext=(-11 if lab == 'P' else 4, -13), textcoords='offset points',
                            color=col, fontsize=10, weight='bold')
        lineas.append((ln, y))
        cursores.append(cur)

    # espectrograma: se va revelando de izquierda a derecha
    ext = [ts[0]-(ts[1]-ts[0])/2, ts[-1]+(ts[1]-ts[0])/2, fs[0], fs[-1]]
    imagen = axs.imshow(np.full_like(db, np.nan), origin='lower', extent=ext, aspect='auto', cmap='viridis',
                        vmin=-rango_db, vmax=0, interpolation='nearest')
    axs.axhline(CF.FMAX_FUENTE, color='white', lw=1.2, ls=(0, (5, 3)))
    axs.axhline(CF.FMAX_FUENTE, color=vc.TINTA, lw=0.6, ls=(0, (5, 3)))
    cur_s = axs.axvline(0, color='white', lw=1.0)
    if nombre != 'F':
        for tt, col in ((t_p, vc.COLOR_P), (t_s, vc.COLOR_S)):
            axs.axvline(tt, color=col, lw=1.4, ls=(0, (4, 2)))
    axs.set_xlim(0, CF.tfin)
    axs.set_ylim(0, fmax)
    axs.set_xlabel('tiempo (s)', color=vc.TINTA_SUAVE)
    axs.set_ylabel('frecuencia (Hz)', color=vc.TINTA_SUAVE, fontsize=9)
    axs.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)
    for s in axs.spines.values():
        s.set_color('#9aa5b1')
    cb = fig.colorbar(imagen, ax=axs, pad=0.01, fraction=0.04)
    cb.set_label('dB', fontsize=8, color=vc.TINTA_SUAVE)
    cb.ax.tick_params(labelsize=8, colors=vc.TINTA_SUAVE)

    # esquema del modelo con la estación
    dx, dy, km = CF.dx, CF.dy, 1e-3
    js = vc.superficie()
    x = np.arange(CF.nx)*dx*km
    axm.fill_between(x, js*dy*km, CF.ny*dy*km, color=vc.AIRE, linewidth=0)
    axm.plot(x, js*dy*km, color=vc.TINTA, lw=1.2)
    for nom, (i, j) in vc.posiciones_estaciones().items():
        if nom == 'F':
            continue
        axm.plot(i*dx*km, j*dy*km, 'o', ms=4, mfc='#9aa5b1', mec='white', mew=0.8)
    fi, fj = vc.TOP.sour1
    axm.plot(fi*dx*km, fj*dy*km, '*', ms=11, mfc=vc.FUENTE, mec='white', mew=0.8)
    marca, = axm.plot(pos[0]*dx*km, pos[1]*dy*km, 'o' if nombre != 'F' else '*', ms=11 if nombre != 'F' else 15,
                      mfc=vc.ESTACION if nombre != 'F' else vc.FUENTE, mec=vc.TINTA, mew=1.5)
    axm.annotate(nombre, (pos[0]*dx*km, pos[1]*dy*km), xytext=(7, 8), textcoords='offset points', fontsize=11, color=vc.TINTA, weight='bold')
    axm.set_xlim(0, CF.nx*dx*km)
    axm.set_ylim(0, CF.ny*dy*km)
    axm.set_aspect('equal')
    axm.set_xlabel('x (km)', color=vc.TINTA_SUAVE, fontsize=8)
    axm.set_ylabel('altura (km)', color=vc.TINTA_SUAVE, fontsize=8)
    axm.tick_params(labelsize=7, colors=vc.TINTA_SUAVE)
    axm.set_title('posición', loc='left', fontsize=9, color=vc.TINTA)
    nombre_largo = 'Fuente (F)' if nombre == 'F' else 'Estación %s' % nombre
    titulo = fig.suptitle('', fontsize=13, color=vc.TINTA, x=0.01, ha='left')

    def cuadro(k):
        tc = cuadros[k]
        n = int(np.searchsorted(t, tc + 1e-9))
        for (ln, y) in lineas:
            ln.set_data(t[:n], y[:n])
        for c in cursores:
            c.set_xdata([tc, tc])
        cur_s.set_xdata([tc, tc])
        visible = np.where(ts[None, :] <= tc, db, np.nan)
        imagen.set_data(visible)
        est = 0 if nombre == 'F' else vc.estado_estacion(nombre, tc)
        if nombre != 'F':
            marca.set_markerfacecolor((vc.ESTACION, vc.COLOR_P, vc.COLOR_S)[est])
        estado = ('', '   ·   llegó la onda P', '   ·   llegó la onda S')[est]
        titulo.set_text('%s   ·   sismograma y espectrograma (%s)   ·   t = %5.2f s%s' % (nombre_largo, componente, tc, estado))
        return imagen, titulo

    if formato == 'mp4' and not shutil.which('ffmpeg'):
        formato = 'gif'
        print('No hay ffmpeg: se guarda como GIF.')
    archivo = os.path.join(carpeta_salida, 'estacion_%s.%s' % (nombre, formato))
    escritor = animation.FFMpegWriter(fps=fps, bitrate=2500) if formato == 'mp4' else animation.PillowWriter(fps=fps)
    ani = animation.FuncAnimation(fig, cuadro, frames=len(cuadros), blit=False)
    ani.save(archivo, writer=escritor, dpi=dpi, savefig_kwargs=dict(facecolor='white'))
    plt.close(fig)
    print('Guardado: %s  (%d cuadros, %.1f s de video)' % (archivo, len(cuadros), len(cuadros)/fps))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--carpeta', default='salida')
    p.add_argument('--salida', default=None)
    p.add_argument('--estaciones', nargs='+', default=None)
    p.add_argument('--componente', choices=['modulo', 'vx', 'vy'], default='modulo')
    p.add_argument('--paso', type=float, default=0.1)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--ventana', type=float, default=2.0)
    p.add_argument('--fmax', type=float, default=2*CF.FMAX_FUENTE)
    p.add_argument('--rango-db', type=float, default=50.)
    p.add_argument('--dpi', type=int, default=90)
    p.add_argument('--formato', choices=['mp4', 'gif'], default='mp4')
    a = p.parse_args()
    nombres, vx, vy = vc.cargar_estaciones(a.carpeta)
    salida = a.salida or os.path.join(a.carpeta, 'animaciones_estaciones')
    os.makedirs(salida, exist_ok=True)
    for nom in (a.estaciones or nombres):
        k = nombres.index(nom)
        animar_estacion(nom, vx[k], vy[k], salida, a.componente, a.paso, a.fps, a.ventana, a.fmax, a.rango_db, a.dpi, a.formato)


if __name__ == '__main__':
    main()
