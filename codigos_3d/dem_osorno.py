# -*- coding: utf-8 -*-
"""
dem_osorno.py: modelo digital de elevación del volcán Osorno en una malla regular.

Fuente: teselas SRTM de 1 arco-segundo (~30 m) del conjunto "Terrain Tiles" alojado en AWS
(https://registry.opendata.aws/terrain-tiles/, formato skadi/HGT): S41W073 y S42W073.
SRTM es un producto de la NASA/NGA; su uso es libre y se pide citarlo.

Genera osorno_dem_50m.npz con:
    z   (ny, nx) cota en metros sobre el nivel del mar, malla de paso dx metros, z[j, i]
                 con j hacia el NORTE e i hacia el ESTE (convención del solver: x = i*dx, y = j*dy)
    dx  paso de malla (m);  lat0, lon0  coordenadas de la cima (centro del dominio)

Uso:   python dem_osorno.py [--lado 25000] [--dx 50] [--descargar]
"""

import argparse, gzip, io, os, urllib.request
import numpy as np

TESELAS = ('S41W073', 'S42W073')          # lat -41..-40 y -42..-41, lon -73..-72
URL = 'https://elevation-tiles-prod.s3.amazonaws.com/skadi/%s/%s.hgt.gz'
N = 3601                                   # muestras por lado de una tesela
M_POR_GRADO = 111195.                      # metros por grado de latitud


def cargar_teselas(carpeta='.', descargar=False):
    z = []
    for t in TESELAS:
        ruta = os.path.join(carpeta, t + '.hgt')
        if not os.path.isfile(ruta):
            if not descargar:
                raise SystemExit('Falta %s: ejecute con --descargar' % ruta)
            datos = urllib.request.urlopen(URL % (t[:3], t), timeout=120).read()
            with open(ruta, 'wb') as f:
                f.write(gzip.decompress(datos))
        z.append(np.fromfile(ruta, dtype='>i2').reshape(N, N).astype(np.float32))
    norte, sur = z
    assert np.array_equal(norte[-1], sur[0]), 'las teselas no coinciden en la costura'
    return np.vstack([norte[:-1], sur])    # fila 0: 40°S ... última: 42°S (lon 73°O ... 72°O)


def remuestrear(mosaico, lado=25000., dx=50.):
    """Malla regular de lado x lado metros centrada en la cima (máximo del mosaico)."""
    i0, j0 = np.unravel_index(mosaico.argmax(), mosaico.shape)
    lat0, lon0 = -40.0 - i0/3600., -73.0 + j0/3600.
    dy_arc = M_POR_GRADO/3600.
    dx_arc = M_POR_GRADO*np.cos(np.radians(lat0))/3600.
    n = int(round(lado/dx)) + 1
    x = (np.arange(n) - (n-1)/2)*dx        # metros hacia el este desde la cima
    y = (np.arange(n) - (n-1)/2)*dx        # metros hacia el norte
    col = j0 + x/dx_arc                    # columna fraccionaria (este = col creciente)
    fila = i0 - y/dy_arc                   # fila fraccionaria (norte = fila decreciente)
    f0 = np.floor(fila).astype(int); c0 = np.floor(col).astype(int)
    wf = (fila - f0)[:, None]; wc = (col - c0)[None, :]
    g = lambda a, b: mosaico[np.ix_(a, b)]
    z = ((1-wf)*(1-wc)*g(f0, c0) + (1-wf)*wc*g(f0, c0+1) + wf*(1-wc)*g(f0+1, c0) + wf*wc*g(f0+1, c0+1))
    return z, lat0, lon0                   # z[j, i]: j=0 es el borde SUR


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lado', type=float, default=25000.)
    ap.add_argument('--dx', type=float, default=50.)
    ap.add_argument('--carpeta', default='.')
    ap.add_argument('--descargar', action='store_true')
    a = ap.parse_args()
    z, lat0, lon0 = remuestrear(cargar_teselas(a.carpeta, a.descargar), a.lado, a.dx)
    # j hacia el norte: invertir filas (la fila 0 del remuestreo es el borde norte)
    z = z[::-1].copy()
    np.savez_compressed('osorno_dem_%dm.npz' % a.dx, z=z.astype(np.float32), dx=a.dx, lat0=lat0, lon0=lon0)
    print('Cima: %.4f, %.4f  |  malla %s, paso %g m  |  cota %.0f a %.0f m' % (lat0, lon0, z.shape, a.dx, z.min(), z.max()))
