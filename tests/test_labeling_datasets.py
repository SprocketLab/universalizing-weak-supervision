import numpy as np
import pandas as pd
import pytest

from uws.datasets import BasicColumnarDataset, BoardGamesRankingDataset, create_dataset
from uws.labeling import ContinuousLF, DiscreteLF, FeatureRankingLF


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"id": np.arange(n), "owned": rng.random(n), "rank": rng.permutation(n),
                       "cat": rng.choice(["a", "b", "c"], n), "minage": rng.integers(0, 18, n)})
    df = pd.concat([df, df.iloc[:3]])  # duplicates are dropped
    p = tmp_path / "items.csv"
    df.to_csv(p, index=False)
    return tmp_path, p


def test_columnar_dataset(csv):
    _, p = csv
    ds = BasicColumnarDataset(p, "id", ["owned", "rank", "id"], label_feature="rank")
    assert len(ds.df) == 60
    fm = ds.get_feature_map(5)
    assert set(fm) == {"owned"}
    assert ds.get_label(5) == ds.df.loc[5, "rank"]


def test_feature_ranking_lf():
    lf = FeatureRankingLF("x", 3, highest_first=True)
    assert lf.apply([{"x": 0.1}, {"x": 0.9}, {"x": 0.5}]).permutation == [1, 2, 0]


def test_regression_lfs(csv):
    _, p = csv
    df = pd.read_csv(p)
    with pytest.warns(UserWarning):
        L = ContinuousLF(["owned", "missing"]).apply(df)
    assert list(L.columns) == ["owned"] and L["owned"].between(0, 10).all()
    L2 = DiscreteLF("cat", "rank").apply(df)  # non-numeric columns are fine
    assert L2["cat"].notna().all()
    zero = pd.DataFrame({"z": np.zeros(5)})
    with pytest.warns(UserWarning):
        assert (ContinuousLF("z").apply(zero)["z"] == 6).all()


def test_ranking_dataset_roundtrip(csv):
    root, p = csv
    conf = {"dataset_name": "boardgames", "project_root": str(root), "raw_data_path": p.name,
            "processed_data_path": "proc", "features": ["owned", "minage", "rank"], "label_feature": "rank",
            "highest_first": False, "n_train": 20, "n_test": 10, "dimension": 4, "seed": 0,
            "recreate_if_exists": True}
    ds = create_dataset(conf)
    assert isinstance(ds, BoardGamesRankingDataset)
    ds.create_samples()
    assert ds.X.shape == (30, 4, 2) and len(ds.Y) == 30
    # cached samples load back (the old code crashed here on a key mismatch)
    ds2 = create_dataset({**conf, "recreate_if_exists": False})
    ds2.create_samples()
    assert [y.permutation for y in ds2.Y] == [y.permutation for y in ds.Y]
    with pytest.raises(ValueError):
        create_dataset({**conf, "dataset_name": "imagenet"})
