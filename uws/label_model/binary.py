"""Binary label model: a thin wrapper around FlyingSquid (requires ``universalizing-weak-supervision[binary]``)."""


def BinaryLabelModel(m: int, **kwargs):
    """FlyingSquid's triplet-based binary label model over ``m`` LFs (votes in {-1, 0, +1})."""
    try:
        from flyingsquid.label_model import LabelModel
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "BinaryLabelModel needs `pip install universalizing-weak-supervision[binary]` (flyingsquid, pgmpy)"
        ) from e
    return LabelModel(m, **kwargs)


__all__ = ["BinaryLabelModel"]
