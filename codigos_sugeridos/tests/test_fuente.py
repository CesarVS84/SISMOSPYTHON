"""La fuente conserva la forma del original y su espectro cae a -40 dB a 8 Hz."""
import numpy as np
import fuente


def r_original(t, h=5., w=1256.5, z=0.0001):
    return w*(1*np.exp((-1+np.sqrt(z))/2*(h*10*t))-1*np.exp((-1-np.sqrt(z))/2*(h*10*t)))


def test_misma_forma_que_el_original():
    # r(t; h) = r_original(t*h/5): solo cambia la escala de tiempo
    t = np.linspace(0, 3, 500)
    assert np.allclose(fuente.r(t), r_original(t*fuente.h1/5.), rtol=1e-12, atol=1e-12)
    assert abs(fuente.r(np.linspace(0, 10, 100000)).max() - r_original(np.linspace(0, 10, 100000)).max()) < 5e-3


def test_espectro_menos_40_db_a_8_hz():
    dt = 0.001
    t = np.arange(0, 80, dt)
    p = fuente.r(t)
    P = np.abs(np.fft.rfft(p))
    f = np.fft.rfftfreq(len(t), dt)
    nivel = 20*np.log10(P/P.max())
    assert nivel[np.argmin(abs(f-8.))] < -39.5
    assert f[np.argmax(nivel <= -40)] <= 8.1
    # y el original tenía mucho más contenido a 8 Hz
    po = np.abs(np.fft.rfft(r_original(t)))
    assert 20*np.log10(po[np.argmin(abs(f-8.))]/po.max()) > -20
