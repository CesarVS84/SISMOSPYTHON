# -*- coding: utf-8 -*-
"""
CPML.py: simulación de un sismo volcánico (elastodinámica 2-D, diferencias
finitas, capa absorbente CPML y topografía).

Se ejecuta con:      python CPML.py            (simulación completa)
                     python CPML.py --rapido   (prueba corta: 300 pasos)

SISMO ANALIZADO: sistema de ecuaciones diferenciales lineales acoplado

    dvx/dt  = 1/rho ( dSxx/dx + dSxy/dy )
    dvy/dt  = 1/rho ( dSxy/dx + dSyy/dy )
    dSxx/dt = (lambda+2mu) dvx/dx + lambda dvy/dy
    dSyy/dt = (lambda+2mu) dvy/dy + lambda dvx/dx
    dSxy/dt = mu ( dvx/dy + dvy/dx )

Discretización (la misma del programa original): velocidades en los nodos
(nx*ny), esfuerzos en el centro de las celdas ((nx-1)*(ny-1)); cada derivada se
calcula en el centro de la celda (o en el nodo) como el promedio de dos
diferencias. En un paso de tiempo primero se actualizan las velocidades con los
esfuerzos vigentes y luego los esfuerzos con las velocidades nuevas.

Qué cambia respecto del programa original (CPML.py de la carpeta superior):
  * Cada paso de tiempo son dos núcleos de Numba en paralelo (velocidad y esfuerzo)
    que incluyen el interior, la capa CPML, la fuente y el borde de la malla. El
    original recorría la malla con bucles de Python y funciones que leían arreglos
    globales (Numba los congela al compilar: usaban siempre los valores del primer
    paso).
  * Los arreglos "viejo/nuevo" se intercambian (ping-pong). Con vx = vx_out ambos
    nombres eran el mismo arreglo y la CPML recibía velocidades ya actualizadas.
  * Las estaciones se guardan en arreglos preasignados (no con np.append).
  * Las instantáneas se guardan en el paso más cercano al tiempo pedido.
  * No se construye el esfuerzo con borde (Sb) salvo al guardar una instantánea.
"""

import os
import sys
import time
import numpy as np
from numba import config, njit, prange, set_num_threads

import CF

# MacBook Air M4: 10 hilos. Si la máquina tiene menos, se usan los disponibles.
set_num_threads(min(CF.NUM_HILOS, config.NUMBA_NUM_THREADS))


# =========================================================================
#  Núcleos
# =========================================================================

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def paso_velocidad(vx_n, vy_n, vx, vy, Sxx, Sxy, Syy, aa, damp, dt, dx, dy,
                   m_xx_dx, m_xy_dx, m_xy_dy, m_yy_dy,
                   m_xx_dx_o, m_xy_dx_o, m_xy_dy_o, m_yy_dy_o,
                   a_x, b_x, k_x, a_y, b_y, k_y,
                   P1, P2, Q1, Q2, nx, ny):
    '''Velocidades nuevas (vx_n, vy_n) a partir de las anteriores (vx, vy) y de
    los esfuerzos vigentes (Sxx, Sxy, Syy: celdas). aa = 1/masa efectiva de cada
    nodo (0 en el aire: esos nodos no se mueven). En la capa se usa la CPML.
    La velocidad es nula en el borde de la malla.'''
    nc = nx-1
    for j in prange(ny):
        en_y = j < Q1 or j >= ny-Q2
        for i in range(nx):
            p = i+nx*j
            a = aa[p]
            if i == 0 or i == nx-1 or j == 0 or j == ny-1 or a == 0.:
                vx_n[p] = 0.
                vy_n[p] = 0.
                continue
            # celdas que rodean al nodo: (i-1,j-1) (i,j-1) (i-1,j) (i,j)
            c00 = i-1+nc*(j-1)
            c10 = i+nc*(j-1)
            c01 = i-1+nc*j
            c11 = i+nc*j
            dsxxdx = (0.5/dx)*(Sxx[c10]-Sxx[c00]+Sxx[c11]-Sxx[c01])
            dsxydx = (0.5/dx)*(Sxy[c10]-Sxy[c00]+Sxy[c11]-Sxy[c01])
            dsxydy = (0.5/dy)*(Sxy[c01]-Sxy[c00]+Sxy[c11]-Sxy[c10])
            dsyydy = (0.5/dy)*(Syy[c01]-Syy[c00]+Syy[c11]-Syy[c10])
            if i < P1 or i >= nx-P2:
                m_xx_dx_o[p] = b_x[p]*m_xx_dx[p] + a_x[p]*dsxxdx
                m_xy_dx_o[p] = b_x[p]*m_xy_dx[p] + a_x[p]*dsxydx
                dsxxdx = dsxxdx/k_x[p] + m_xx_dx_o[p]
                dsxydx = dsxydx/k_x[p] + m_xy_dx_o[p]
            if en_y:
                m_xy_dy_o[p] = b_y[p]*m_xy_dy[p] + a_y[p]*dsxydy
                m_yy_dy_o[p] = b_y[p]*m_yy_dy[p] + a_y[p]*dsyydy
                dsxydy = dsxydy/k_y[p] + m_xy_dy_o[p]
                dsyydy = dsyydy/k_y[p] + m_yy_dy_o[p]
            vx_n[p] = vx[p]*(1.-damp) + dt*a*(dsxxdx+dsxydy)
            vy_n[p] = vy[p]*(1.-damp) + dt*a*(dsxydx+dsyydy)


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def paso_esfuerzo(Sxx_n, Sxy_n, Syy_n, Sxx, Sxy, Syy, vx, vy, b, c, d, dt, dx, dy,
                  m_vx_dx, m_vy_dx, m_vx_dy, m_vy_dy,
                  m_vx_dx_o, m_vy_dx_o, m_vx_dy_o, m_vy_dy_o,
                  a_xh, b_xh, k_xh, a_yh, b_yh, k_yh,
                  P1, P2, Q1, Q2, nx, ny):
    '''Esfuerzos nuevos de cada celda a partir de los anteriores y de las
    velocidades nuevas (vx, vy). b = lambda+2mu, c = lambda, d = mu por celda
    (0 en el aire: el esfuerzo allí es nulo). En la capa se usa la CPML.'''
    nc = nx-1
    for j in prange(ny-1):
        en_y = j < Q1 or j >= ny-1-Q2
        for i in range(nx-1):
            q = i+nc*j
            if b[q] == 0.:
                Sxx_n[q] = 0.
                Sxy_n[q] = 0.
                Syy_n[q] = 0.
                continue
            k = i+nx*j
            dvxdx = (0.5/dx)*(vx[k+1]-vx[k]+vx[k+1+nx]-vx[k+nx])
            dvydx = (0.5/dx)*(vy[k+1]-vy[k]+vy[k+1+nx]-vy[k+nx])
            dvxdy = (0.5/dy)*(vx[k+nx]-vx[k]+vx[k+1+nx]-vx[k+1])
            dvydy = (0.5/dy)*(vy[k+nx]-vy[k]+vy[k+1+nx]-vy[k+1])
            if i < P1 or i >= nx-1-P2:
                m_vx_dx_o[q] = b_xh[q]*m_vx_dx[q] + a_xh[q]*dvxdx
                m_vy_dx_o[q] = b_xh[q]*m_vy_dx[q] + a_xh[q]*dvydx
                dvxdx = dvxdx/k_xh[q] + m_vx_dx_o[q]
                dvydx = dvydx/k_xh[q] + m_vy_dx_o[q]
            if en_y:
                m_vx_dy_o[q] = b_yh[q]*m_vx_dy[q] + a_yh[q]*dvxdy
                m_vy_dy_o[q] = b_yh[q]*m_vy_dy[q] + a_yh[q]*dvydy
                dvxdy = dvxdy/k_yh[q] + m_vx_dy_o[q]
                dvydy = dvydy/k_yh[q] + m_vy_dy_o[q]
            Sxx_n[q] = Sxx[q] + dt*(b[q]*dvxdx + c[q]*dvydy)
            Syy_n[q] = Syy[q] + dt*(b[q]*dvydy + c[q]*dvxdx)
            Sxy_n[q] = Sxy[q] + dt*d[q]*(dvxdy + dvydx)


@njit(cache=True)
def aplicar_fuente(vx, vy, fuente50, fuente90, rc, rs, nx):
    '''Impone la velocidad de la fuente en los nodos F == 50 (+) y F == 90 (-).'''
    for n in range(fuente50.shape[0]):
        p = fuente50[n, 0] + nx*fuente50[n, 1]
        vx[p] = rc
        vy[p] = -rs
    for n in range(fuente90.shape[0]):
        p = fuente90[n, 0] + nx*fuente90[n, 1]
        vx[p] = -rc
        vy[p] = rs


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def construir_Sb(Sbxx, Sbyy, Sbxy, Sxx, Syy, Sxy, nx, ny):
    '''Esfuerzos con borde (nx+1)*(ny+1): Sb[i, j] = S[i-1, j-1]; contorno nulo.'''
    for j in prange(1, ny):
        for i in range(1, nx):
            Sbxx[i+(nx+1)*j] = Sxx[i-1+(nx-1)*(j-1)]
            Sbyy[i+(nx+1)*j] = Syy[i-1+(nx-1)*(j-1)]
            Sbxy[i+(nx+1)*j] = Sxy[i-1+(nx-1)*(j-1)]


# =========================================================================
#  Instantáneas: (tiempo, sufijo del nombre del archivo), como en el original
# =========================================================================
INSTANTANEAS = (
    (0.1, '1e1'), (0.2, '2e1'), (0.3, '3e1'), (0.4, '4e1'), (0.5, '5e1'), (0.6, '6e1'),
    (0.7, '7e1'), (0.8, '8e1'), (0.9, '9e1'), (1, '1'), (1.1, '11'), (1.2, '12'),
    (1.3, '13'), (1.4, '14'), (1.5, '15'), (1.6, '16'), (1.7, '17'), (1.8, '18'),
    (1.9, '19'), (2, '2'), (2.102, '21'), (2.2, '22'), (2.3, '23'), (2.4, '24'),
    (2.5, '25'), (2.6, '26'), (2.7, '27'), (2.8, '28'), (2.9, '29'), (3, '3'),
    (3.3, '33'), (3.2, '32'), (3.4, '34'), (3.5, '35'), (3.6, '36'), (3.7, '37'),
    (3.8, '38'), (3.9, '39'), (4, '4'), (4.1, '41'), (4.2, '42'), (4.3, '43'),
    (4.4, '44'), (4.5, '45'), (4.6, '46'), (4.7, '47'), (4.8, '48'), (4.9, '49'),
    (5, '5'), (5.1, '51'), (5.2, '52'), (5.3, '53'), (5.4, '54'), (5.5, '55'),
    (5.6, '56'), (5.7, '57'), (5.8, '58'), (5.9, '59'), (6, '6'), (6.1, '61'),
    (6.2, '62'), (6.3, '63'), (6.4, '64'), (6.5, '65'), (6.6, '66'), (6.7, '67'),
    (6.8, '68'), (6.9, '69'), (7.005, '7'), (7.1, '71'), (7.2, '72'), (7.3, '73'),
    (7.4, '74'), (7.5, '75'), (7.6, '76'), (7.705, '77'), (7.8, '78'), (7.9, '79'),
    (8, '8'), (8.1, '81'), (8.2, '82'), (8.3, '83'), (8.4, '84'), (8.5, '85'),
    (8.6, '86'), (8.7, '87'), (8.8, '88'), (8.9, '89'), (9, '9'), (9.105, '91'),
    (9.2, '92'), (9.3, '93'), (9.4, '94'), (9.5, '95'), (9.6, '96'), (9.7, '97'),
    (9.8, '98'), (9.9, '99'), (10, '10'), (10.1, '101'), (10.2, '102'), (10.3, '103'),
    (10.4, '104'), (10.505, '105'), (10.6, '106'), (10.7, '107'), (10.8, '108'), (10.9, '109'),
    (11, '110'), (11.1, '1101'), (11.2, '1102'), (11.3, '1103'), (11.4, '1104'), (11.5, '1105'),
    (11.6, '1106'), (11.7, '1107'), (11.8, '1108'), (11.9, '1109'), (12, '120'), (12.1, '1201'),
    (12.2, '1202'), (12.3, '1203'), (12.4, '1204'), (12.5, '1205'), (12.605, '1206'), (12.7, '1207'),
    (12.8, '1208'), (12.9, '1209'), (13, '130'), (13.1, '1301'), (13.2, '1302'), (13.3, '1303'),
    (13.4, '1304'), (13.5, '1305'), (13.6, '1306'), (13.7, '1307'), (13.8, '1308'), (13.9, '1309'),
    (14.005, '140'), (14.1, '1401'), (14.2, '1402'), (14.3, '1403'), (14.4, '1404'), (14.5, '1405'),
    (14.6, '1406'), (14.705, '1407'), (14.8, '1408'), (14.9, '1409'),
)


# =========================================================================
#  Simulación
# =========================================================================

def simular(modelo, perfiles, fuente_r, estaciones, nx=CF.nx, ny=CF.ny,
            dx=CF.dx, dy=CF.dy, dt=CF.dt, nt=CF.nt, damp=CF.damp,
            theta=np.radians(CF.thetag), P=(CF.pml_points_x1, CF.pml_points_x2,
                                            CF.pml_points_y1, CF.pml_points_y2),
            usar_jih=CF.USAR_JIH, carpeta=None, instantaneas=INSTANTANEAS,
            esfuerzo_inicial=None, verbose=True, registrar_campo=None):
    """Avanza nt pasos de tiempo.

    modelo : dict con aa (nx*ny), lambdaa, mu (celdas), F (nx, ny) y, si usar_jih,
             too ((nx+1)*(ny+1)) y jih = lista de pasadas (elevation, bandera, si, sj).
    perfiles : dict de CBA.crear_perfiles().
    fuente_r : función r(t) de la fuente (fuente.py).
    estaciones : lista de [i, j]; la última se guarda como "fa" (fuente).
    Devuelve el registro de velocidad de las estaciones, [estación, paso, (vx, vy)]."""
    P1, P2, Q1, Q2 = P
    nn = nx*ny
    nc = (nx-1)*(ny-1)
    aa = np.ascontiguousarray(modelo['aa'], dtype=np.float64)
    c = np.ascontiguousarray(modelo['lambdaa'], dtype=np.float64)
    d = np.ascontiguousarray(modelo['mu'], dtype=np.float64)
    b = c + 2*d
    fuente50 = np.argwhere(np.asarray(modelo['F']) == 50).astype(np.int64)
    fuente90 = np.argwhere(np.asarray(modelo['F']) == 90).astype(np.int64)

    vx, vy, vx_n, vy_n = (np.zeros(nn) for _ in range(4))
    Sxx, Sxy, Syy, Sxx_n, Sxy_n, Syy_n = (np.zeros(nc) for _ in range(6))
    if esfuerzo_inicial is not None:
        w, six, siy, sixy = esfuerzo_inicial
        Sxx[w] = Sxx_n[w] = six
        Syy[w] = Syy_n[w] = siy
        Sxy[w] = Sxy_n[w] = sixy
    mv, mv_o = [np.zeros(nc) for _ in range(4)], [np.zeros(nc) for _ in range(4)]   # vx_dx, vy_dx, vx_dy, vy_dy
    ms, ms_o = [np.zeros(nn) for _ in range(4)], [np.zeros(nn) for _ in range(4)]   # xx_dx, xy_dx, xy_dy, yy_dy
    Sbxx, Sbyy, Sbxy = (np.zeros((nx+1)*(ny+1)) for _ in range(3))
    Unewx, Unewy = np.zeros(nn), np.zeros(nn)
    if usar_jih:
        from CBO import cboveljih
        vxc, vyc = np.zeros(nn), np.zeros(nn)
        too, pasadas = modelo['too'], modelo['jih']
        vp, vs = CF.vp, CF.vs

    registro = np.zeros((len(estaciones), nt+1, 2))
    idx_est = np.array([i+nx*j for i, j in estaciones], dtype=np.int64)
    paso_guardado = {}
    if carpeta is not None:
        os.makedirs(carpeta, exist_ok=True)
    if carpeta is not None and instantaneas is not None:
        paso_guardado = {int(round(tg/dt)): suf for tg, suf in instantaneas if tg <= nt*dt + 1e-9}
    tipo = np.float32 if CF.SNAPSHOT_FLOAT32 else np.float64

    def guardar(sufijo, vxs, vys):
        np.save(os.path.join(carpeta, 'vyt'+sufijo), vys.astype(tipo))
        np.save(os.path.join(carpeta, 'vxt'+sufijo), vxs.astype(tipo))
        if CF.GUARDAR_ESFUERZOS:
            construir_Sb(Sbxx, Sbyy, Sbxy, Sxx, Syy, Sxy, nx, ny)   # esfuerzos vigentes al inicio del paso
            np.save(os.path.join(carpeta, 'sxxt'+sufijo), Sbxx.astype(tipo))
            np.save(os.path.join(carpeta, 'syyt'+sufijo), Sbyy.astype(tipo))
            np.save(os.path.join(carpeta, 'sxyt'+sufijo), Sbxy.astype(tipo))

    t_inicio = time.time()
    for n in range(nt+1):
        t = n*dt
        # Fuente: se impone sobre las velocidades del paso anterior
        rt = fuente_r(t)
        aplicar_fuente(vx, vy, fuente50, fuente90, rt*np.cos(theta), rt*np.sin(theta), nx)

        paso_velocidad(vx_n, vy_n, vx, vy, Sxx, Sxy, Syy, aa, damp, dt, dx, dy,
                       ms[0], ms[1], ms[2], ms[3], ms_o[0], ms_o[1], ms_o[2], ms_o[3],
                       perfiles['a_x'], perfiles['b_x'], perfiles['k_x'],
                       perfiles['a_y'], perfiles['b_y'], perfiles['k_y'],
                       P1, P2, Q1, Q2, nx, ny)

        if usar_jih:
            # Borde topográfico (Jih): lee una copia para que los hilos no se pisen
            vxc[:] = vx_n
            vyc[:] = vy_n
            for elevation, flag, si, sj in pasadas:
                cboveljih(elevation, flag, too, vxc, vyc, vx_n, vy_n, vp, vs, dx, dy, nx, ny, si, sj)

        registro[:, n, 0] = vx_n[idx_est]
        registro[:, n, 1] = vy_n[idx_est]
        if registrar_campo is not None:
            registrar_campo(n, vx_n, vy_n)

        paso_esfuerzo(Sxx_n, Sxy_n, Syy_n, Sxx, Sxy, Syy, vx_n, vy_n, b, c, d, dt, dx, dy,
                      mv[0], mv[1], mv[2], mv[3], mv_o[0], mv_o[1], mv_o[2], mv_o[3],
                      perfiles['a_x_half'], perfiles['b_x_half'], perfiles['k_x_half'],
                      perfiles['a_y_half'], perfiles['b_y_half'], perfiles['k_y_half'],
                      P1, P2, Q1, Q2, nx, ny)

        Unewx += vx_n*dt
        Unewy += vy_n*dt

        if n in paso_guardado:
            guardar(paso_guardado[n], vx_n, vy_n)
        if carpeta is not None and n == nt:
            guardar('fin', vx_n, vy_n)
            np.save(os.path.join(carpeta, 'Dfin'), Unewx.astype(tipo))
            np.save(os.path.join(carpeta, 'Dfiny'), Unewy.astype(tipo))

        if verbose and n % 250 == 0:
            print('paso %5d de %d   t = %6.3f s   transcurrido: %6.1f s' % (n, nt, t, time.time()-t_inicio), flush=True)

        # Intercambio de arreglos: lo "nuevo" pasa a ser lo vigente
        vx, vx_n = vx_n, vx
        vy, vy_n = vy_n, vy
        Sxx, Sxx_n = Sxx_n, Sxx
        Sxy, Sxy_n = Sxy_n, Sxy
        Syy, Syy_n = Syy_n, Syy
        mv, mv_o = mv_o, mv
        ms, ms_o = ms_o, ms

    if verbose:
        print('Simulación terminada en %.1f s (%.2f ms por paso)' % (time.time()-t_inicio, 1e3*(time.time()-t_inicio)/(nt+1)))
    return registro


def guardar_estaciones(registro, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    for k in range(registro.shape[0]-1):
        np.save(os.path.join(carpeta, 'estacion%dvx' % (k+1)), registro[k, :, 0])
        np.save(os.path.join(carpeta, 'estacion%dvy' % (k+1)), registro[k, :, 1])
    np.save(os.path.join(carpeta, 'favx'), registro[-1, :, 0])
    np.save(os.path.join(carpeta, 'favy'), registro[-1, :, 1])


def main(rapido=False, carpeta='salida'):
    import TOP
    import prueba
    import CBA
    import fuente

    CF.informe()
    print('Hilos de Numba:', config.NUMBA_NUM_THREADS, '(en uso:', min(CF.NUM_HILOS, config.NUMBA_NUM_THREADS), ')')
    print('Fuente: h = %.3f, frecuencia máxima (%.0f dB) = %.2f Hz' % (
        fuente.h1, CF.NIVEL_FUENTE_DB, fuente.frecuencia_maxima(fuente.h1)))

    assert CF.dx == CF.dy, 'El esquema supone dx == dy'
    # velocidad P máxima del modelo (para la CPML)
    cp = float(np.sqrt(((TOP.lambdaa + 2*TOP.mu)[TOP.mu > 0]/CF.Rho).max()))
    perfiles = CBA.crear_perfiles(cp=cp)

    # sj = -1: el modelo tiene j hacia arriba; si = -1: segunda pasada para los tramos que bajan
    modelo = dict(aa=TOP.aa, lambdaa=TOP.lambdaa, mu=TOP.mu, F=TOP.F, too=TOP.too,
                  jih=[(prueba.elevation, prueba.boundary_flag_velocity, 1, -1),
                       (prueba.elevation_izq, prueba.boundary_flag_velocity_izq, -1, -1)])
    estaciones = list(TOP.estaciones) + [TOP.sour1]
    esf = None
    if CF.USAR_ESFUERZO_INICIAL:
        esf = ((CF.nx-1)*175 + 200, CF.six, CF.siy, CF.sixy)

    nt = 300 if rapido else CF.nt
    registro = simular(modelo, perfiles, fuente.r, estaciones, nt=nt, carpeta=carpeta,
                       esfuerzo_inicial=esf, instantaneas=None if rapido else INSTANTANEAS)
    guardar_estaciones(registro, carpeta)
    print('Resultados en la carpeta:', os.path.abspath(carpeta))
    return registro


if __name__ == '__main__':
    main(rapido='--rapido' in sys.argv)
