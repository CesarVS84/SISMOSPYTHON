"""La fuente conserva la forma del original y su espectro cae a CF.NIVEL_FUENTE_DB a 8 Hz."""
import CF
import numpy as np
import fuente


def r_original(t, h=5., w=1256.5, z=0.0001):
    return w*(1*np.exp((-1+np.sqrt(z))/2*(h*10*t))-1*np.exp((-1-np.sqrt(z))/2*(h*10*t)))


def test_misma_forma_que_el_original():
    # r(t; h) = r_original(t*h/5): solo cambia la escala de tiempo
    t = np.linspace(0, 3, 500)
    assert np.allclose(fuente.r(t), r_original(t*fuente.h1/5.), rtol=1e-12, atol=1e-12)
    assert abs(fuente.r(np.linspace(0, 10, 100000)).max() - r_original(np.linspace(0, 10, 100000)).max()) < 5e-3


def test_espectro_cae_al_nivel_a_8_hz():
    dt = 0.001
    t = np.arange(0, 80, dt)
    p = fuente.r(t)
    P = np.abs(np.fft.rfft(p))
    f = np.fft.rfftfreq(len(t), dt)
    nivel = 20*np.log10(P/P.max())
    assert abs(nivel[np.argmin(abs(f-CF.FMAX_FUENTE))] - CF.NIVEL_FUENTE_DB) < 0.6
    assert f[np.argmax(nivel <= CF.NIVEL_FUENTE_DB)] <= CF.FMAX_FUENTE + 0.1
    # y el original tenía mucho más contenido a 8 Hz
    po = np.abs(np.fft.rfft(r_original(t)))
    assert 20*np.log10(po[np.argmin(abs(f-8.))]/po.max()) > CF.NIVEL_FUENTE_DB + 3


def test_tasa_de_momento_integra_m0_y_parte_de_cero():
    dt = 0.0005
    t = np.arange(0, 8, dt)
    m = fuente.tasa_momento(t)
    assert abs(m.sum()*dt/CF.M0 - 1) < 1e-3
    assert m[0] < 1e-6*m.max()
    assert abs(m[-1]) < 1e-6*m.max()


def test_velocidad_radiada_bajo_el_nivel_a_8_hz():
    dt = 0.0005
    t = np.arange(0, 8, dt)
    v = np.gradient(fuente.tasa_momento(t), dt)          # onda radiada ~ derivada de la tasa
    V = np.abs(np.fft.rfft(v, 2**17))
    f = np.fft.rfftfreq(2**17, dt)
    banda = (f > 7.5) & (f < 8.5)
    assert 20*np.log10(V[banda].mean()/V.max()) < CF.NIVEL_FUENTE_DB + 0.5
    assert abs(v.sum()*dt) < 1e-3*CF.M0 / fuente.sigma_t  # bipolar: sin desplazamiento permanente


def test_doble_cupla_sin_traza_ni_isotropia():
    mxx, myy, mxy = fuente.tensor_unitario(30.)
    assert abs(mxx+myy) < 1e-12                           # traza nula: cizalle puro
    assert abs(mxx**2 + myy**2 + 2*mxy**2 - 2) < 1e-12    # norma de una doble cupla unitaria


def test_inyeccion_de_momento_reparte_con_pesos_que_suman_uno():
    import CPML
    nx, ny = 40, 40
    b = np.ones((nx-1)*(ny-1))
    b[:5] = 0.                                      # unas celdas de aire no reciben fuente
    F = np.zeros((nx, ny)); F[20, 20] = 50; F[21, 21] = 90
    idx, peso = CPML.crear_fuente_momento(F, b, 1.5, nx, ny)
    assert abs(peso.sum() - 1) < 1e-12 and np.all(b[idx] > 0)
    S = [np.zeros((nx-1)*(ny-1)) for _ in range(3)]
    CPML.inyectar_momento(S[0], S[1], S[2], idx, peso, 2., 3., 5.)
    assert np.allclose([s.sum() for s in S], [-2., -3., -5.])
