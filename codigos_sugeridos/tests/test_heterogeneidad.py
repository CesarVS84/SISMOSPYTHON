"""La heterogeneidad es reproducible, solo toca la roca y respeta la estabilidad."""
import numpy as np
import CF
import heterogeneidad as H
import TOP


def test_campo_tiene_media_0_y_desviacion_1_y_es_suave():
    e = H.campo_aleatorio((200, 200), 300., CF.dx, 3)
    assert abs(e.mean()) < 1e-9 and abs(e.std() - 1) < 1e-9
    # suave: la diferencia entre celdas vecinas es mucho menor que la amplitud
    assert np.abs(np.diff(e, axis=1)).std() < 0.5


def test_solo_cambia_la_roca_y_es_reproducible(monkeypatch):
    monkeypatch.setattr(CF, 'HETEROGENEIDAD', 0.2)
    lam1, mu1 = H.perturbar(TOP.lambdaa, TOP.mu)
    lam2, mu2 = H.perturbar(TOP.lambdaa, TOP.mu)
    assert np.array_equal(mu1, mu2)
    aire = TOP.mu == 0
    assert np.all(mu1[aire] == 0) and np.all(lam1[aire] == 0)
    assert np.all(mu1[~aire] != TOP.mu[~aire])
    assert abs(np.std(mu1[~aire]/TOP.mu[~aire]) - 0.2) < 0.03
    # la razón lambda/mu se conserva (se multiplican por el mismo factor)
    assert np.allclose(lam1[~aire]/mu1[~aire], TOP.lambdaa[~aire]/TOP.mu[~aire])


def test_courant_bajo_el_limite_con_la_heterogeneidad_por_defecto():
    lam, mu = H.perturbar(TOP.lambdaa, TOP.mu)
    cp = np.sqrt(((lam+2*mu)[mu > 0]/CF.Rho).max())
    assert cp*CF.dt/min(CF.dx, CF.dy) < 0.95


def test_desactivada_no_modifica_nada(monkeypatch):
    monkeypatch.setattr(CF, 'HETEROGENEIDAD', 0.)
    lam, mu = H.perturbar(TOP.lambdaa, TOP.mu)
    assert lam is TOP.lambdaa and mu is TOP.mu
