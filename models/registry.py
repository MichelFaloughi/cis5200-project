"""
Model factory utilities.

The evaluation notebooks can use `get_model("model3")` (or any alias)
without worrying about where a particular implementation lives inside
`models/`.
"""

from typing import Any, Dict, Type

from .model0 import PersistenceModel
from .model1 import LinearRegressionModel
from .model2 import RandomForestModel
from .model3 import XGBoostModel
from .model4 import MLPModel
from .model5 import TransformerModel

MODEL_REGISTRY: Dict[str, Type[Any]] = {
    "model0": PersistenceModel,
    "baseline": PersistenceModel,
    "persistence": PersistenceModel,
    "model1": LinearRegressionModel,
    "linear": LinearRegressionModel,
    "linear_regression": LinearRegressionModel,
    "model2": RandomForestModel,
    "random_forest": RandomForestModel,
    "rf": RandomForestModel,
    "model3": XGBoostModel,
    "xgboost": XGBoostModel,
    "model4": MLPModel,
    "mlp": MLPModel,
    "model5": TransformerModel,
    "transformer": TransformerModel,
}


def get_model(name: str, **kwargs):
    """
    Instantiate one of the registered models.

    Args:
        name: Registry key (case-insensitive). Supports aliases such as
              'baseline', 'xgboost', 'mlp', etc.
        **kwargs: Keyword arguments forwarded to the model constructor.
    """
    key = name.lower()
    try:
        model_cls = MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from exc
    return model_cls(**kwargs)


def available_models():
    """Return the sorted list of unique model identifiers."""
    return sorted(set(MODEL_REGISTRY))

