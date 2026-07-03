"""Tabular teacher used to anchor the GNN as a graph residual.

The teacher is trained on the GNN's ``train_mask`` accounts only, so its score is a
non-leaky feature for validation/test accounts. Train accounts receive out-of-fold
(cross-fit) predictions so the GNN never sees optimistic in-sample teacher scores.

The teacher probability is converted to a logit and added to the GNN's class-1 logit
during training and inference:

    out[:, 1] = out[:, 1] + alpha * teacher_logit
"""

from typing import Dict, Tuple

import numpy as np

TEACHER_MODELS = ("numpy_logistic", "logistic", "random_forest", "gradient_boosting")


def _logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(prob.astype(np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _fit_core(
    x: np.ndarray,
    y: np.ndarray,
    model_type: str,
    class_weight: str,
    lr: float,
    max_iter: int,
    l2: float,
    seed: int,
    n_estimators: int = 200,
    max_depth: int = None,
) -> Dict:
    """Fit a teacher and return a picklable payload consumable by ``_predict_core``."""
    if model_type == "numpy_logistic":
        # Deferred import: train_baseline_model -> evaluate -> gnn_inference imports
        # this module, so a top-level import here would be circular.
        from .train_baseline_model import _fit_logistic_regression

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        xs = ((x - mean) / std).astype(np.float32)
        weights, bias = _fit_logistic_regression(xs, y.astype(np.int64), class_weight, lr, max_iter, l2)
        return {"type": "numpy_logistic", "mean": mean, "std": std, "weights": weights, "bias": bias}

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            f"scikit-learn is required for teacher model `{model_type}`. "
            "Use `--teacher-model numpy_logistic` or install requirements."
        ) from exc

    cw = None if class_weight == "none" else "balanced"
    if model_type == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight=cw, max_iter=max_iter, random_state=seed),
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, class_weight=cw, random_state=seed
        )
    elif model_type == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=0.1, max_depth=max_depth or 3, random_state=seed
        )
    else:
        raise RuntimeError(f"Unknown teacher model: {model_type}")
    estimator.fit(x, y)
    return {"type": model_type, "estimator": estimator}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _predict_core(payload: Dict, x: np.ndarray) -> np.ndarray:
    if payload["type"] == "numpy_logistic":
        xs = ((x - payload["mean"]) / payload["std"]).astype(np.float32)
        return _sigmoid(xs @ payload["weights"] + payload["bias"])
    return payload["estimator"].predict_proba(x)[:, 1]


def compute_teacher_logits(
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    *,
    model_type: str = "numpy_logistic",
    folds: int = 5,
    class_weight: str = "balanced",
    lr: float = 0.05,
    max_iter: int = 1000,
    l2: float = 1e-4,
    seed: int = 42,
    n_estimators: int = 200,
    max_depth: int = None,
) -> Tuple[np.ndarray, Dict]:
    """Return (teacher_logit for every account, full-train payload for inference).

    Non-train accounts get full-train predictions; train accounts get K-fold
    out-of-fold predictions to avoid in-sample leakage into the GNN residual.
    """
    if model_type not in TEACHER_MODELS:
        raise RuntimeError(f"Unknown teacher model: {model_type}")
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise RuntimeError("Tabular teacher needs at least one training account.")

    x_train = x_all[train_idx]
    y_train = y_all[train_idx].astype(np.int64)
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("Tabular teacher needs both classes in the training slice.")

    full_payload = _fit_core(x_train, y_train, model_type, class_weight, lr, max_iter, l2, seed, n_estimators, max_depth)
    prob = _predict_core(full_payload, x_all).astype(np.float64)

    if folds >= 2 and train_idx.size >= folds:
        rng = np.random.default_rng(seed)
        order = np.arange(train_idx.size)
        rng.shuffle(order)
        for fold_positions in np.array_split(order, folds):
            if fold_positions.size == 0:
                continue
            rest_positions = np.setdiff1d(np.arange(train_idx.size), fold_positions, assume_unique=False)
            y_rest = y_train[rest_positions]
            if len(set(y_rest.tolist())) < 2:
                continue  # keep the full-train prediction as a fallback
            fold_payload = _fit_core(
                x_train[rest_positions], y_rest, model_type, class_weight, lr, max_iter, l2, seed, n_estimators, max_depth
            )
            hold_global = train_idx[fold_positions]
            prob[hold_global] = _predict_core(fold_payload, x_all[hold_global]).astype(np.float64)

    return _logit(prob), full_payload


def apply_teacher_logits(payload: Dict, x_all: np.ndarray) -> np.ndarray:
    """Recompute teacher logits at inference time from a stored payload."""
    return _logit(_predict_core(payload, x_all).astype(np.float64))
