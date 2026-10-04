# -*- coding: utf-8 -*-
"""
visual_comun.py: funciones compartidas por graficas.py, espectrograma.py y
animacion.py (lectura de los resultados de CPML.py y dibujo del modelo).

Los resultados se leen de la carpeta que escribe `python CPML.py` (por defecto
'salida'): instantáneas vxt<sufijo>.npy / vyt<sufijo>.npy y los registros de las
estaciones estacion<k>vx.npy, estacion<k>vy.npy, favx.npy, favy.npy.
"""

import os
import numpy as np
import matplotlib
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

import CF
import TOP
from CPML import INSTANTANEAS

# Colores (superficies claras, texto en tinta neutra; la identidad la dan el marcador y la etiqueta)
TINTA = '#1f2933'
TINTA_SUAVE = '#52606d'
AIRE = '#e4e7eb'
CPML_COLOR = '#52606d'
ESTACION = '#0b7285'          # estación: la onda aún no llega
COLOR_P = '#f08c00'           # llegó la onda P
COLOR_S = '#c2255c'           # llegó la onda S
FUENTE = '#c92a2a'
CMAP_COMPONENTE = 'RdBu_r'      # divergente: azul / blanco (cero) / rojo
CMAP_MODULO = 'viridis'         # secuencial: de menor a mayor módulo

NOMBRES_ESTACIONES = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']


def posiciones_estaciones():
    """{nombre: (i, j)} de las estaciones y de la fuente 'F'."""
    d = {NOMBRES_ESTACIONES[k]: tuple(p) for k, p in enumerate(TOP.estaciones)}
    d['F'] = tuple(TOP.sour1)
    return d


def estacion_superficie_mas_cercana():
    """Nombre de la estación sobre la superficie libre más cercana a la fuente."""
    js = superficie()
    fi, fj = TOP.sour1
    mejor, dmin = None, np.inf
    for k, (i, j) in enumerate(TOP.estaciones):
        if j == js[i]:                                  # solo estaciones en la superficie
            d = np.hypot(i-fi, j-fj)
            if d < dmin:
                mejor, dmin = NOMBRES_ESTACIONES[k], d
    return mejor


def distancia_a_la_fuente(nombre):
    """Distancia en línea recta (m) entre la estación y la fuente."""
    i, j = posiciones_estaciones()[nombre]
    fi, fj = TOP.sour1
    return float(np.hypot((i-fi)*CF.dx, (j-fj)*CF.dy))


def tiempos_llegada(nombre):
    """(t_P, t_S) teóricos en s: distancia en línea recta / vp y / vs. Para la fuente (F)
    la distancia es cero. No incluyen el retraso de la fuente (la tasa de momento tiene su máximo en ~0.3 s)."""
    r = distancia_a_la_fuente(nombre)
    return r/CF.vp, r/CF.vs


def serie(vx, vy, componente):
    """Serie temporal de 'vx', 'vy' o 'modulo'."""
    if componente == 'vx':
        return vx
    if componente == 'vy':
        return vy
    return np.hypot(vx, vy)


def stft(x, dt, ventana, solape):
    """STFT con ventana de Hann. Devuelve (t centro de cada ventana, f, |X|).
    La señal se rellena (por reflexión) media ventana a cada lado, de modo que los centros de las
    ventanas cubren desde t = 0 hasta el final (sin el rellenado la primera ventana completa
    quedaría centrada en ventana/2). En los bordes la ventana ve en parte una copia reflejada de la señal."""
    nv = int(round(ventana/dt))
    salto = max(1, int(round(nv*(1-solape))))
    w = np.hanning(nv)
    x = x - x.mean()
    x = np.pad(x, nv//2, mode='reflect')        # reflexión: sin escalón en los bordes
    pos = np.arange(0, len(x)-nv+1, salto)
    nfft = 1 << int(np.ceil(np.log2(4*nv)))              # relleno con ceros: curvas más suaves
    X = np.array([np.abs(np.fft.rfft(w*x[p:p+nv], nfft)) for p in pos]).T
    f = np.fft.rfftfreq(nfft, dt)
    return pos*dt, f, X/w.sum()*2


def espectrograma_db(x, dt, ventana=2.0, solape=0.9, fmax=16., rango_db=50., ref=None):
    """Espectrograma en dB respecto de su máximo (o de ref). Devuelve (t, f, dB) con f <= fmax."""
    t, f, X = stft(x, dt, ventana, solape)
    ref = X.max() if ref is None else ref
    db = 20*np.log10(np.maximum(X, 1e-30)/ref)
    sel = f <= fmax
    return t, f[sel], np.maximum(db[sel], -rango_db)


def superficie():
    """Fila de la superficie en cada columna."""
    if hasattr(TOP, 'js'):
        return np.asarray(TOP.js)
    return np.asarray(TOP.hc).astype(int)[:CF.nx]


_MMAP = {}


def _campos_mmap(carpeta):
    """(vx, vy, tiempos) del formato nuevo (campo_vx.npy, campo_vy.npy, tiempos.npy) o None."""
    if carpeta not in _MMAP:
        a = os.path.join(carpeta, 'campo_vx.npy')
        if os.path.exists(a):
            _MMAP[carpeta] = (np.load(a, mmap_mode='r'), np.load(os.path.join(carpeta, 'campo_vy.npy'), mmap_mode='r'),
                              np.load(os.path.join(carpeta, 'tiempos.npy')))
        else:
            _MMAP[carpeta] = None
    return _MMAP[carpeta]


def instantaneas_disponibles(carpeta):
    """Lista [(tiempo, id)] de las instantáneas. id es el número de instantánea (formato
    nuevo, cada CF.SNAPSHOT_CADA s) o el sufijo del nombre del archivo (formato antiguo)."""
    nuevo = _campos_mmap(carpeta)
    if nuevo is not None:
        return [(float(t), k) for k, t in enumerate(nuevo[2])]
    out = [(t, s) for t, s in INSTANTANEAS if os.path.exists(os.path.join(carpeta, 'vxt%s.npy' % s))]
    if os.path.exists(os.path.join(carpeta, 'vxtfin.npy')):
        out.append((CF.tfin, 'fin'))
    if not out:
        raise SystemExit("No hay instantáneas en '%s'. Ejecute antes `python CPML.py`." % carpeta)
    return sorted(out)


def cargar_campo(carpeta, ident):
    """(vx, vy) como arreglos [j, i] (j hacia arriba)."""
    nuevo = _campos_mmap(carpeta)
    if nuevo is not None and isinstance(ident, (int, np.integer)):
        return np.array(nuevo[0][ident], dtype=float), np.array(nuevo[1][ident], dtype=float)
    vx = np.load(os.path.join(carpeta, 'vxt%s.npy' % ident)).astype(float).reshape(CF.ny, CF.nx)
    vy = np.load(os.path.join(carpeta, 'vyt%s.npy' % ident)).astype(float).reshape(CF.ny, CF.nx)
    return vx, vy


def elegir_instantanea(carpeta, t):
    disp = instantaneas_disponibles(carpeta)
    k = int(np.argmin([abs(t-ti) for ti, _ in disp]))
    return disp[k]


def cargar_estaciones(carpeta):
    """Registros de las estaciones: (nombres, vx[k, paso], vy[k, paso])."""
    vx, vy, nombres = [], [], []
    for k in range(1, 10):
        a = os.path.join(carpeta, 'estacion%dvx.npy' % k)
        if os.path.exists(a):
            vx.append(np.load(a))
            vy.append(np.load(os.path.join(carpeta, 'estacion%dvy.npy' % k)))
            nombres.append(NOMBRES_ESTACIONES[k-1])
    if os.path.exists(os.path.join(carpeta, 'favx.npy')):
        vx.append(np.load(os.path.join(carpeta, 'favx.npy')))
        vy.append(np.load(os.path.join(carpeta, 'favy.npy')))
        nombres.append('F')
    if not vx:
        raise SystemExit("No hay registros de estaciones en '%s'." % carpeta)
    return nombres, np.array(vx), np.array(vy)


def escala_campo(valores, vmax, gamma, con_signo):
    """Transforma el campo para el color: (v/vmax)^gamma conservando el signo.
    gamma < 1 realza las ondas débiles (la amplitud cae mucho con la distancia)."""
    r = np.clip(valores/vmax, -1 if con_signo else 0, 1)
    return np.sign(r)*np.abs(r)**gamma if con_signo else r**gamma


def estado_estacion(nombre, t):
    """0: la onda aún no llega; 1: ya llegó la P; 2: ya llegó la S (tiempos teóricos)."""
    tp, ts = tiempos_llegada(nombre)
    return 2 if t >= ts else (1 if t >= tp else 0)


def actualizar_marcas(ax, t):
    """Cambia el color de las estaciones del mapa según lleguen la onda P y la S."""
    for nombre, ln in ax._marcas.items():
        e = estado_estacion(nombre, t)
        ln.set_markerfacecolor((ESTACION, COLOR_P, COLOR_S)[e])
        ln.set_markersize(9.5 if (e and min(t-tiempos_llegada(nombre)[e-1], 9) < 0.25) else 6.5)   # destello al llegar


def dibujar_mapa(ax, campo, vmax, gamma=0.5, con_signo=True, titulo='', estaciones=True,
                 etiquetas=True, t=None):
    """Dibuja el campo [j, i] con la topografía, el aire, la capa CPML y las estaciones.
    Devuelve la imagen (para actualizarla en la animación)."""
    nx, ny, dx, dy = CF.nx, CF.ny, CF.dx, CF.dy
    km = 1e-3
    ext = [-dx/2*km, (nx-0.5)*dx*km, -dy/2*km, (ny-0.5)*dy*km]
    cmap = CMAP_COMPONENTE if con_signo else CMAP_MODULO
    im = ax.imshow(escala_campo(campo, vmax, gamma, con_signo), origin='lower', extent=ext,
                   cmap=cmap, vmin=-1 if con_signo else 0, vmax=1, interpolation='nearest',
                   aspect='equal', zorder=1)

    # aire (sobre la superficie): gris neutro, para no confundir "aire" con "campo cero"
    js = superficie()
    x = np.arange(nx)*dx*km
    ys = js*dy*km
    ax.fill_between(x, ys, ext[3], color=AIRE, zorder=2, linewidth=0)
    ax.plot(x, ys, color=TINTA, lw=1.4, zorder=4)

    # capa CPML
    rect = [(ext[0], P*dx*km - dx/2*km, ext[2], ext[3]) for P in [CF.pml_points_x1]]
    rect.append(((nx-CF.pml_points_x2-0.5)*dx*km, ext[1], ext[2], ext[3]))
    if CF.pml_points_y1 > 0:
        rect.append((ext[0], ext[1], ext[2], (CF.pml_points_y1-0.5)*dy*km))
    if CF.pml_points_y2 > 0:
        rect.append((ext[0], ext[1], (ny-CF.pml_points_y2-0.5)*dy*km, ext[3]))
    for x0, x1, y0, y1 in rect:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, hatch='///', edgecolor=CPML_COLOR,
                               linewidth=0.0, alpha=0.35, zorder=3))
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=CPML_COLOR, linewidth=1.0,
                               linestyle=(0, (4, 3)), zorder=5))

    if estaciones:
        halo = [pe.withStroke(linewidth=2.5, foreground='white')]
        ax._marcas = {}
        for k, (i, j) in enumerate(TOP.estaciones):
            ln, = ax.plot(i*dx*km, j*dy*km, 'o', ms=6.5, mfc=ESTACION, mec='white', mew=1.3, zorder=7)
            ax._marcas[NOMBRES_ESTACIONES[k]] = ln
            if etiquetas:
                ax.annotate(NOMBRES_ESTACIONES[k], (i*dx*km, j*dy*km), xytext=(4, 7), textcoords='offset points',
                            fontsize=8, color=TINTA, path_effects=halo, zorder=8)
        i, j = TOP.sour1
        ax.plot(i*dx*km, j*dy*km, '*', ms=13, mfc=FUENTE, mec='white', mew=1.0, zorder=7)
        if etiquetas:
            ax.annotate('F', (i*dx*km, j*dy*km), xytext=(6, -11), textcoords='offset points', fontsize=8,
                        color=TINTA, path_effects=halo, zorder=8)

    if estaciones and t is not None:
        actualizar_marcas(ax, t)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel('x (km)', color=TINTA_SUAVE)
    ax.set_ylabel('altura j·dy (km)', color=TINTA_SUAVE)
    ax.set_title(titulo, fontsize=11, color=TINTA, loc='left')
    ax.tick_params(colors=TINTA_SUAVE, labelsize=8)
    for s in ax.spines.values():
        s.set_color('#9aa5b1')
    return im


def barra_color(fig, im, ax, vmax, gamma, con_signo, etiqueta):
    """Barra de color con ticks en valores reales (m/s) aunque el color esté en escala gamma."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    valores = np.array([-1, -.25, -.0625, 0, .0625, .25, 1])*vmax if con_signo else np.array([0, .0625, .25, .5, 1])*vmax
    pos = escala_campo(valores, vmax, gamma, con_signo)
    cb.set_ticks(pos)
    cb.set_ticklabels(['%.2g' % v for v in valores])
    cb.ax.tick_params(labelsize=8, colors=TINTA_SUAVE)
    cb.set_label(etiqueta, fontsize=9, color=TINTA_SUAVE)
    cb.outline.set_edgecolor('#9aa5b1')
    return cb


def leyenda_modelo(fig, y=0.01):
    """Leyenda común: aire, capa CPML, estación, fuente."""
    elementos = [
        Patch(facecolor=AIRE, edgecolor='none', label='Aire'),
        Patch(facecolor='none', edgecolor=CPML_COLOR, hatch='///', label='Capa CPML (absorbente)'),
        Line2D([], [], color=TINTA, lw=1.4, label='Superficie libre (topografía)'),
        Line2D([], [], marker='o', ls='', mfc=ESTACION, mec='white', ms=7, label='Estación (aún sin onda)'),
        Line2D([], [], marker='o', ls='', mfc=COLOR_P, mec='white', ms=7, label='llegó la onda P'),
        Line2D([], [], marker='o', ls='', mfc=COLOR_S, mec='white', ms=7, label='llegó la onda S'),
        Line2D([], [], marker='*', ls='', mfc=FUENTE, mec='white', ms=12, label='Fuente (F)'),
    ]
    fig.legend(handles=elementos, loc='lower center', ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, y), labelcolor=TINTA_SUAVE)


def campo_derivado(vx, vy, campo):
    """(valores, con_signo, etiqueta) para 'vx', 'vy' o 'modulo'."""
    if campo == 'vx':
        return vx, True, 'vx (m/s)'
    if campo == 'vy':
        return vy, True, 'vy (m/s)'
    return np.hypot(vx, vy), False, '|v| (m/s)'


def vmax_robusto(campos, p=99.7):
    """Escala del color: percentil p del |campo| (un máximo aislado de la fuente no aplasta el resto)."""
    return float(np.percentile(np.abs(np.concatenate([c.ravel() for c in campos])), p)) or 1.0
