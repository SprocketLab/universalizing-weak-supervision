"""Common interface of UWS label models (snorkel-style ``fit`` / ``predict``)."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseLabelModel(ABC):
    """Base class: ``fit`` on unlabeled LF outputs, ``predict`` latent labels, ``get_weights`` per LF."""

    #: per-LF canonical parameters (e.g. Mallows theta, accuracy), set by ``fit``
    thetas: Any = None

    @abstractmethod
    def fit(self, L, *args, **kwargs) -> BaseLabelModel:
        """Estimate LF parameters from unlabeled LF outputs ``L`` (examples x LFs)."""

    @abstractmethod
    def predict(self, L, *args, **kwargs):
        """Infer the latent label of every example in ``L``."""

    def get_weights(self) -> np.ndarray:
        """Learned per-LF weights (canonical parameters)."""
        if self.thetas is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted yet")
        return np.asarray(self.thetas)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> BaseLabelModel:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} holds a {type(obj).__name__}, not a {cls.__name__}")
        return obj
