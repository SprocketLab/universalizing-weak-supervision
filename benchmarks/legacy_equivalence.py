import sys, numpy as np, random, warnings
warnings.simplefilter('ignore')
OLD = sys.argv[1]
sys.path.insert(0, OLD)
import ranking_utils as oru, ws_ranking as ows, ws_regression as oreg
sys.path.pop(0)
import mallows as omal
_dlp, _ddlp = omal.Mallows.d_log_partition, omal.Mallows.d_d_log_partition
omal.Mallows.d_log_partition = staticmethod(lambda t, d: _dlp(float(np.squeeze(t)), d))  # old code + numpy>=2
omal.Mallows.d_d_log_partition = staticmethod(lambda t, d: _ddlp(float(np.squeeze(t)), d))
from uws.ranking import Ranking, RankingUtils, RankingLabelModel, sample_mallows_LFs, generate_true_rankings
from uws.regression import ContinuousLabelModel, generate_lfs, generate_synthetic_xy

rng = np.random.default_rng(0)
for d, p in [(5, None), (6, 0.6), (8, 0.4)]:
    ru = RankingUtils(d, legacy=True); oru_u = oru.RankingUtils(d)
    Y = generate_true_rankings(d, 300, rng, ru)
    thetas = [0.3, 1.5, 0.6, 2.0, 0.4, 1.0]
    L = sample_mallows_LFs(Y, 6, thetas, p=p, rng=rng)
    # rebuild identical old-code rankings (same permutation and mask)
    Lo = []
    for row in L:
        orow = []
        for r in row:
            o = oru.Ranking(list(r.permutation), oru_u); o.mask = r.mask.copy(); o.z = oru_u.ranking_to_pair_signs(o); orow.append(o)
        Lo.append(orow)
    assert all(np.array_equal(a.z, b.z) for ra, rb in zip(L, Lo) for a, b in zip(ra, rb))
    dist_new = [ru.kendall_tau_distance(L[i][0], L[i][1]) for i in range(50)]
    dist_old = [oru_u.kendall_tau_distance(Lo[i][0], Lo[i][1]) for i in range(50)]
    print(f"d={d} p={p}: kendall max diff {np.max(np.abs(np.array(dist_new)-dist_old)):.2e}", end="; ")
    for tm in ["triplet", "triplet_opt", "median_triplet", "median_triplet_opt"]:
        w = ows.WeakSupRanking(oru_u); w.train({"train_method": tm}, Lo)
        lm = RankingLabelModel(ru, train_method=tm, theta_bounds=(1e-1, 100) if tm.endswith('opt') else (-np.inf, np.inf)).fit(L)
        print(f"{tm} theta diff {np.max(np.abs(w.thetas - lm.thetas)):.1e}", end="; ")
    for rule in ["kemeny", "weighted_kemeny"]:
        yo = w.infer_ranking({"inference_rule": rule}, Lo)
        yn = lm.predict(L, inference_rule=rule)
        wts = None if rule == "kemeny" else lm.thetas
        def cost(i, y):
            w_ = np.ones(6) if wts is None else wts
            return np.mean([w_[j] * ru.kendall_tau_distance(y, L[i][j]) for j in range(6) if L[i][j].permutation != y.permutation])
        same = np.mean([a.permutation == b.permutation for a, b in zip(yo, yn)])
        tie = np.max([abs(cost(i, yn[i]) - cost(i, Ranking(yo[i].permutation, ru) if False else yn[i].__class__(yo[i].permutation, ru, z=yo[i].z))) for i in range(len(L))])
        print(f"{rule} identical {same:.3f} (max objective gap {tie:.1e})", end="; ")
    print()

X, Yr, v = generate_synthetic_xy(2000, 10, rng)
Lr, _ = generate_lfs(6, 2000, Yr, v)
o = oreg.LabelModel(); o.fit(Lr, v, median=True)
o.Y_hat = np.array([(o.Sigma_hat[6, :6] @ np.linalg.inv(o.Sigma_hat[:6, :6]) @ Lr[i]) for i in range(len(Lr))])  # old inference, minus the numpy-2 crash
n = ContinuousLabelModel(legacy=True).fit(Lr, v, median=True)
print("regression legacy (median) max |diff| in predictions:", np.max(np.abs(o.Y_hat - n.predict(Lr))))
