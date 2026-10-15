import numpy as np
import pytest


def test_geodesic_label_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("geoopt")
    from geoopt.manifolds.lorentz import Lorentz

    from uws.label_model import GeodesicLabelModel
    from uws.synthetic import LFGenerator, generate_data, sample_params

    torch.manual_seed(0)
    man = Lorentz(k=1.0)
    p, beta = sample_params(man, 5)
    _, _, y = generate_data(man, 100, p, beta, sig=0.2)
    lfs = LFGenerator(man, 6, siglb=0.1, sigub=0.6, heterogeneous=False, rng=np.random.default_rng(0)).fit().predict(y)
    lm = GeodesicLabelModel(man).fit(lfs)
    y_hat = lm.predict(lfs)
    assert torch.isfinite(y_hat).all()
    best_single = min(float(man.dist2(lf, y).mean()) for lf in lfs)
    assert float(man.dist2(y_hat, y).mean()) < best_single
    legacy = GeodesicLabelModel(man, legacy=True).fit(lfs)
    assert len(legacy.thetas) == 6
    with pytest.raises(ValueError, match="inf/NaN"):  # overflowing distances fail loudly
        far = LFGenerator(man, 3, siglb=30, sigub=40, heterogeneous=False, rng=np.random.default_rng(0))
        GeodesicLabelModel(man).fit(far.fit().predict(y))


def test_ptranking_vendored_import():
    pytest.importorskip("torch")
    from uws._vendor.ptranking.ltr_adhoc.eval.ltr import LTREvaluator  # noqa: F401


def test_snorkel_ranking_baseline():
    pytest.importorskip("snorkel")
    from uws.experiments.real_workflow import snorkel_rankings
    from uws.ranking import RankingUtils
    from uws.synthetic import generate_true_rankings, sample_mallows_LFs

    rng = np.random.default_rng(0)
    ru = RankingUtils(3)
    Y = generate_true_rankings(3, 100, rng, ru)
    L = sample_mallows_LFs(Y, 3, [1.0, 2.0, 3.0], rng=rng)
    assert len(snorkel_rankings(L, ru, n_epochs=50)) == 100
