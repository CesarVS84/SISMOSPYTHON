"""La conversión 2-D -> 3-D reproduce la fórmula de campo lejano 3-D (Aki y Richards 4.29)."""
import numpy as np
import CF
import fuente
import solucion_analitica as SA
import conversion_2d3d as C


def test_semiderivada_aplicada_dos_veces_es_la_derivada():
    # el operador es no causal con cola t^(-3/2): se usa una ventana larga para que la cola
    # truncada no pese
    dt = 0.01
    t = np.arange(0, 400, dt)
    x = np.exp(-0.5*((t-200)/0.4)**2)
    d = C.semiderivada(C.semiderivada(x, dt), dt)
    ref = np.gradient(x, dt)
    assert np.abs(d-ref).max() < 5e-3*np.abs(ref).max()


def test_conversion_reproduce_la_onda_p_3d_en_campo_lejano():
    dt = 0.005
    r = 60000.                                         # k r >> 1 (a 3 Hz, k r = 100)
    mxx, myy, mxy = fuente.tensor_unitario()
    phi = 0.3                                          # dirección de observación (rad)
    x, y = r*np.cos(phi), r*np.sin(phi)
    t, vx2, vy2 = SA.velocidad(x, y, fuente.tasa_momento, dt, CF.Rho, CF.vp, CF.vs, 0., mxx, myy, mxy, T=120.)
    g = np.array([np.cos(phi), np.sin(phi)])
    M = np.array([[mxx, mxy], [mxy, myy]])
    patron = g @ M @ g                                 # gamma_j M_jk gamma_k
    # fórmula 3-D (Aki y Richards 4.29) para el desplazamiento radial de la onda P; velocidad = derivada
    n = len(t)
    u3 = patron*fuente.tasa_momento(t - r/CF.vp)/(4*np.pi*CF.Rho*CF.vp**3*r)     # con M0 = 1 N m en la tasa
    v3 = np.gradient(u3, dt)
    # traza 2-D radial (la P es radial) convertida, con L = 1 (M0_2D = M0_3D)
    v2_rad = vx2*g[0] + vy2*g[1]
    v3_conv = C.traza_3d_equivalente(v2_rad, dt, r, CF.vp, longitud_falla=1.)
    ventana = (t > r/CF.vp-1) & (t < r/CF.vp+3)
    a, b = v3_conv[ventana], v3[ventana]
    assert abs(np.vdot(b, a)/np.vdot(b, b) - 1) < 0.03
    assert np.linalg.norm(a-b)/np.linalg.norm(b) < 0.05
