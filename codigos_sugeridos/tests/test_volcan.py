"""El volcán sintético con los parámetros por defecto: sin NaN, sin crecimiento."""
import numpy as np
import CF, TOP, prueba, CBA, CPML, fuente


def test_volcan_estable():
    cp = float(np.sqrt(((TOP.lambdaa+2*TOP.mu)[TOP.mu > 0]/CF.Rho).max()))
    perf = CBA.crear_perfiles(cp=cp)
    modelo = dict(aa=TOP.aa, lambdaa=TOP.lambdaa, mu=TOP.mu, F=TOP.F, too=TOP.too)
    est = list(TOP.estaciones)+[TOP.sour1]
    reg = CPML.simular(modelo, perf, fuente.r, est, nt=600, verbose=False)
    assert not np.isnan(reg).any()
    a = np.hypot(reg[..., 0], reg[..., 1])
    assert a.max() < 50                       # la fuente impone ~9 m/s
    assert a[:9].max() > 1e-3                 # y la onda llega a las estaciones
