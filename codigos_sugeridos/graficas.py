# -*- coding: utf-8 -*-
"""
graficas.py: mapa de la onda sísmica en un tiempo dado.

Muestra las velocidades (vx, vy y módulo |v|) sobre el modelo, con la topografía,
el aire, la capa absorbente CPML y las estaciones.

Uso (después de `python CPML.py`):
    python graficas.py --t 3.5
    python graficas.py --t 2 4 6 8 --campo vy
    python graficas.py --t 5 --campo modulo --escala lineal --salida onda_t5.png

Opciones:
    --t            tiempo(s) en segundos; se usa la instantánea más cercana
    --campo        todos (vx, vy y |v|; por defecto) | vx | vy | modulo
    --carpeta      carpeta con los resultados (por defecto 'salida')
    --salida       archivo de imagen; con varios tiempos se añade _t<tiempo>
    --escala       raiz (realza las ondas débiles; por defecto) | lineal
    --sin-estaciones
"""

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import visual_comun as vc


def figura(carpeta, t, campo='todos', escala='raiz', estaciones=True):
    ti, suf = vc.elegir_instantanea(carpeta, t)
    vx, vy = vc.cargar_campo(carpeta, suf)
    gamma = 0.5 if escala == 'raiz' else 1.0
    cols = ['vx', 'vy', 'modulo'] if campo == 'todos' else [campo]
    datos = [vc.campo_derivado(vx, vy, c) for c in cols]
    # escala común a las componentes (así vx y vy se comparan); el módulo usa la suya
    vmax_comp = vc.vmax_robusto([vx, vy])
    vmax_mod = vc.vmax_robusto([datos[-1][0]]) if not datos[-1][1] else None

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6.4*n+0.8, 6.9), squeeze=False, constrained_layout=True)
    for ax, (valores, con_signo, etiqueta) in zip(axes[0], datos):
        vmax = vmax_comp if con_signo else vc.vmax_robusto([valores])
        im = vc.dibujar_mapa(ax, valores, vmax, gamma, con_signo, titulo=etiqueta.split(' ')[0], estaciones=estaciones, t=ti)
        vc.barra_color(fig, im, ax, vmax, gamma, con_signo, etiqueta)
    fig.suptitle('Onda sísmica en t = %.2f s' % ti, fontsize=14, color=vc.TINTA, x=0.01, ha='left')
    vc.leyenda_modelo(fig, y=-0.075)
    return fig, ti


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--t', type=float, nargs='+', default=[3.0])
    p.add_argument('--campo', choices=['todos', 'vx', 'vy', 'modulo'], default='todos')
    p.add_argument('--carpeta', default='salida')
    p.add_argument('--salida', default=None)
    p.add_argument('--escala', choices=['raiz', 'lineal'], default='raiz')
    p.add_argument('--sin-estaciones', action='store_true')
    a = p.parse_args()
    for t in a.t:
        fig, ti = figura(a.carpeta, t, a.campo, a.escala, not a.sin_estaciones)
        if a.salida is None:
            nombre = os.path.join(a.carpeta, 'onda_t%05.2f.png' % ti)
        elif len(a.t) == 1:
            nombre = a.salida
        else:
            base, ext = os.path.splitext(a.salida)
            nombre = '%s_t%05.2f%s' % (base, ti, ext or '.png')
        fig.savefig(nombre, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('Guardado:', nombre)


if __name__ == '__main__':
    main()
