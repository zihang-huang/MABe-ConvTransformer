"""
Lightweight XGBoost baseline wired into the existing data pipeline.

The model trains one binary classifier per behavior on frame-level features
emitted by the datasets (raw or precomputed). Training is CPU-only and uses
MultiOutputClassifier for a simple one-vs-rest setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

# Local imports are intentionally inside functions to avoid circular imports
# when the Lightning models are imported elsewhere.


def _to_numpy(array) -> np.ndarray:
    """Detach torch tensors to CPU numpy arrays."""
    if array is None:
        return None
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


@dataclass
class XGBoostSettings:
    """Training-time settings pulled from the Hydra/OmegaConf config."""

    params: Dict
    frame_subsample: int = 1
    max_windows: Optional[int] = None
    max_windows_val: Optional[int] = None
    threshold: float = 0.35


class XGBoostBehaviorModel:
    """
    One-vs-rest XGBoost classifier for frame-level behavior recognition.

    The model expects flattened per-frame features (feature_dim,) and
    binary labels per behavior. Window-level predictions are produced by
    running the classifier on each valid frame in the window.
    """

    def __init__(
        self,
        behaviors: Sequence[str],
        feature_dim: int,
        window_size: int,
        params: Optional[Dict] = None,
        threshold: float = 0.35,
    ):
        self.behaviors = list(behaviors)
        self.num_classes = len(self.behaviors)
        self.feature_dim = feature_dim
        self.window_size = window_size
        base_params = {
            "n_estimators": 300,
            "learning_rate": 0.08,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "objective": "binary:logistic",
            "base_score": 0.5,
        }
        if params:
            base_params.update(params)
        self.model = MultiOutputClassifier(xgb.XGBClassifier(**base_params))
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        payload = {
            "behaviors": self.behaviors,
            "feature_dim": self.feature_dim,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "model": self.model,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostBehaviorModel":
        payload = joblib.load(path)
        obj = cls(
            behaviors=payload["behaviors"],
            feature_dim=int(payload["feature_dim"]),
            window_size=int(payload["window_size"]),
            params=None,
            threshold=float(payload.get("threshold", 0.35)),
        )
        obj.model = payload["model"]
        return obj

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        if y.ndim != 2 or y.shape[1] != self.num_classes:
            raise ValueError(f"Expected y shape (?, {self.num_classes}), got {y.shape}")
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict per-frame probabilities for each behavior.
        Returns array of shape (n_samples, num_classes).
        """
        proba_list = self.model.predict_proba(X)
        # MultiOutputClassifier returns a list of (n_samples, 2) arrays
        probs = np.stack([p[:, 1] if p.ndim > 1 else p for p in proba_list], axis=-1)
        return probs.astype(np.float32, copy=False)

    def predict_window(
        self,
        window_features: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run the classifier on each valid frame of a window.
        Returns (window_size, num_classes) array with zeros for invalid frames.
        """
        feats = np.asarray(window_features)
        if feats.ndim != 2 or feats.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected window features (T, {self.feature_dim}), got {feats.shape}")

        mask = np.ones(feats.shape[0], dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
        if mask.shape[0] != feats.shape[0]:
            raise ValueError(f"valid_mask length {mask.shape[0]} != time dimension {feats.shape[0]}")

        if mask.any():
            probs_valid = self.predict_proba(feats[mask])
            full = np.zeros((feats.shape[0], self.num_classes), dtype=np.float32)
            full[mask] = probs_valid
        else:
            full = np.zeros((feats.shape[0], self.num_classes), dtype=np.float32)
        return full


# ---------------------------------------------------------------------- #
# Data helpers
# ---------------------------------------------------------------------- #
def _sample_indices(total: int, max_windows: Optional[int]) -> Iterable[int]:
    indices = np.arange(total)
    if max_windows is not None and max_windows < total:
        return np.random.choice(indices, size=max_windows, replace=False)
    return indices


def collect_frames_from_dataset(
    dataset,
    frame_subsample: int = 1,
    max_windows: Optional[int] = None,
    desc: str = "dataset",
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a windowed dataset into per-frame feature/label arrays.
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    indices = _sample_indices(len(dataset), max_windows)
    iterator = tqdm(indices, desc=f"[xgboost] Collecting {desc}", disable=not show_progress)
    for idx in iterator:
        sample = dataset[int(idx)]
        feats = _to_numpy(sample["features"])
        labels = _to_numpy(sample.get("labels"))
        mask = _to_numpy(sample.get("valid_mask"))
        mask = np.ones(feats.shape[0], dtype=bool) if mask is None else mask.astype(bool)

        feats = feats[mask]
        labels = labels[mask] if labels is not None else None

        if frame_subsample > 1:
            feats = feats[::frame_subsample]
            labels = labels[::frame_subsample] if labels is not None else None

        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(feats.astype(np.float32, copy=False))
        if labels is not None:
            labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            # Clamp to [0,1] to avoid invalid base_score in XGBoost
            labels = np.clip(labels, 0.0, 1.0)
            y_list.append(labels.astype(np.float32, copy=False))

    if not X_list or not y_list:
        raise ValueError(f"No samples collected from {desc}; check precomputed data and config.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def train_xgboost_from_config(config: Dict) -> Tuple[XGBoostBehaviorModel, Dict]:
    """
    End-to-end training entrypoint used by scripts/train.py when
    config.model.name == 'xgboost'.
    """
    from src.data.dataset import MABeDataModule  # Local import to avoid circular dependency

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    eval_cfg = config.get("evaluation", {})
    paths = config["paths"]

    settings = XGBoostSettings(
        params=model_cfg.get("xgboost_params", {}),
        frame_subsample=int(model_cfg.get("frame_subsample", 1)),
        max_windows=model_cfg.get("max_windows"),
        max_windows_val=model_cfg.get("max_windows_val"),
        threshold=float(model_cfg.get("threshold", eval_cfg.get("threshold", 0.35))),
    )

    features_cfg = config.get("features", {})
    use_engineered = features_cfg.get("use_engineered_features", False)

    dm = MABeDataModule(
        data_dir=paths["data_dir"],
        behaviors=config.get("behaviors", None),
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        target_fps=data_cfg["target_fps"],
        val_split=data_cfg.get("val_split", 0.2),
        test_split=data_cfg.get("test_split", 0.0),
        enable_test_split=data_cfg.get("enable_test_split", False),
        tracking_cache_size=data_cfg.get("tracking_cache_size", 4),
        annotation_cache_size=data_cfg.get("annotation_cache_size", 8),
        use_precomputed=data_cfg.get("use_precomputed", False),
        precomputed_dir=data_cfg.get("precomputed_dir", None),
        shard_cache_size=data_cfg.get("shard_cache_size", 8),
        oversample_rare=data_cfg.get("oversample_rare", False),
        rare_behaviors=data_cfg.get("rare_behaviors", None),
        oversample_factor=data_cfg.get("oversample_factor", 1),
        # Engineered features
        use_engineered_features=use_engineered,
        feature_config=features_cfg,
    )
    dm.setup(stage="fit", use_precomputed=data_cfg.get("use_precomputed", False), precomputed_dir=data_cfg.get("precomputed_dir", None))

    train_ds = dm.train_dataset
    val_ds = dm.val_dataset

    # Disable augmentations for deterministic tree training
    for ds in (train_ds, val_ds):
        if hasattr(ds, "augment"):
            ds.augment = False
        if hasattr(ds, "apply_augment"):
            ds.apply_augment = False

    behaviors = getattr(train_ds, "behaviors", None) or dm.behaviors
    feature_dim = getattr(train_ds, "feature_dim", None)
    window_size = getattr(train_ds, "window_size", data_cfg["window_size"])
    if feature_dim is None:
        sample = train_ds[0]
        feature_dim = _to_numpy(sample["features"]).shape[-1]

    print("[xgboost] Collecting training frames...")
    X_train, y_train = collect_frames_from_dataset(
        train_ds,
        frame_subsample=settings.frame_subsample,
        max_windows=settings.max_windows,
        desc="train",
        show_progress=True,
    )
    print(f"[xgboost] Training data: {X_train.shape[0]} frames, {X_train.shape[1]} features.")

    print("[xgboost] Collecting validation frames...")
    X_val, y_val = collect_frames_from_dataset(
        val_ds,
        frame_subsample=settings.frame_subsample,
        max_windows=settings.max_windows_val,
        desc="val",
        show_progress=True,
    )
    print(f"[xgboost] Validation data: {X_val.shape[0]} frames.")

    model = XGBoostBehaviorModel(
        behaviors=behaviors,
        feature_dim=feature_dim,
        window_size=window_size,
        params=settings.params,
        threshold=settings.threshold,
    )
    print(f"[xgboost] Fitting {model.num_classes} one-vs-rest classifiers...")
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)
    val_pred = (val_probs >= model.threshold).astype(int)
    macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_val, val_pred, average="micro", zero_division=0)
    print(f"[xgboost] Validation macro F1={macro_f1:.4f}, micro F1={micro_f1:.4f}")

    metrics = {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1)}
    return model, metrics


def predict_windows_from_loader(
    model: XGBoostBehaviorModel,
    dataloader,
    keep_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate per-frame probabilities and labels from a dataloader.
    Mirrors collect_predictions() for Lightning models.
    """
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    window_predictions: List[Dict] = []

    for batch in dataloader:
        feats = _to_numpy(batch["features"])  # (B, T, F)
        labels = _to_numpy(batch.get("labels"))
        mask = _to_numpy(batch.get("valid_mask"))

        batch_size, time_dim, _ = feats.shape
        mask = np.ones((batch_size, time_dim), dtype=bool) if mask is None else mask.astype(bool)

        batch_probs = np.zeros((batch_size, time_dim, model.num_classes), dtype=np.float32)
        for i in range(batch_size):
            batch_probs[i] = model.predict_window(feats[i], valid_mask=mask[i])

        if labels is not None:
            valid_mask = mask.reshape(-1)
            all_probs.append(batch_probs.reshape(-1, model.num_classes)[valid_mask])
            all_labels.append(labels.reshape(-1, model.num_classes)[valid_mask])

        if keep_windows:
            video_ids = _to_numpy(batch["video_id"])
            agent_ids = _to_numpy(batch["agent_id"])
            target_ids = _to_numpy(batch["target_id"])
            start_frames = _to_numpy(batch["start_frame"])
            for i in range(batch_size):
                window_predictions.append(
                    {
                        "video_id": int(video_ids[i]),
                        "agent_id": f"mouse{int(agent_ids[i])}" if isinstance(agent_ids[i], (int, np.integer)) else str(agent_ids[i]),
                        "target_id": (
                            f"mouse{int(target_ids[i])}" if isinstance(target_ids[i], (int, np.integer)) and target_ids[i] >= 0 else "self"
                        ),
                        "start_frame": int(start_frames[i]),
                        "probabilities": batch_probs[i],
                    }
                )

    y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, model.num_classes))
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.empty_like(y_probs)
    return y_true, y_probs, window_predictions
