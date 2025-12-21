"""
Temporal feature extraction for MABe behavior recognition.
Computes rolling statistics and temporal context features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


class TemporalFeatureExtractor:
    """
    Extract temporal context features using sliding windows.

    Features extracted:
    - Rolling statistics (mean, std, min, max)
    - Motion energy over windows
    - Trajectory curvature
    - Change point indicators
    """

    def __init__(
        self,
        window_sizes: List[int] = [5, 15, 30, 60],
        fps: float = 30.0
    ):
        """
        Args:
            window_sizes: List of window sizes in frames
            fps: Frame rate
        """
        self.window_sizes = window_sizes
        self.fps = fps

    def extract_features(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute temporal features over multiple window sizes.

        Args:
            features: Input features, shape (n_frames, n_features)
            feature_names: Names of input features (for output naming)

        Returns:
            Dictionary of temporal feature arrays
        """
        n_frames, n_features = features.shape
        temporal_features = {}

        if feature_names is None:
            feature_names = [f'f{i}' for i in range(n_features)]

        # For each window size
        for window in self.window_sizes:
            # Rolling mean
            rolling_mean = self._rolling_mean(features, window)
            for i, name in enumerate(feature_names):
                temporal_features[f'{name}_mean_{window}'] = rolling_mean[:, i]

            # Rolling std
            rolling_std = self._rolling_std(features, window)
            for i, name in enumerate(feature_names):
                temporal_features[f'{name}_std_{window}'] = rolling_std[:, i]

            # Rolling min/max
            rolling_min = self._rolling_min(features, window)
            rolling_max = self._rolling_max(features, window)
            for i, name in enumerate(feature_names):
                temporal_features[f'{name}_min_{window}'] = rolling_min[:, i]
                temporal_features[f'{name}_max_{window}'] = rolling_max[:, i]
                temporal_features[f'{name}_range_{window}'] = rolling_max[:, i] - rolling_min[:, i]

        # Motion energy (requires velocity-like features)
        motion_energy = self._compute_motion_energy(features)
        for window in self.window_sizes:
            temporal_features[f'motion_energy_{window}'] = self._rolling_sum(
                motion_energy, window
            )

        # Rate of change (first derivative features)
        rate_of_change = self._compute_rate_of_change(features)
        for i, name in enumerate(feature_names):
            temporal_features[f'{name}_delta'] = rate_of_change[:, i]

        # Change point detection
        change_points = self._detect_change_points(features)
        temporal_features['change_point_score'] = change_points

        return temporal_features

    def extract_feature_vector(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract temporal features as concatenated array.

        Returns:
            Feature array of shape (n_frames, n_temporal_features)
        """
        temporal_features = self.extract_features(features, feature_names)
        return np.stack(list(temporal_features.values()), axis=-1)

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        return uniform_filter1d(data, size=window, axis=0, mode='nearest')

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        mean = self._rolling_mean(data, window)
        sq_diff = (data - mean) ** 2
        variance = uniform_filter1d(sq_diff, size=window, axis=0, mode='nearest')
        return np.sqrt(variance + 1e-8)

    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling minimum."""
        from scipy.ndimage import minimum_filter1d
        return minimum_filter1d(data, size=window, axis=0, mode='nearest')

    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling maximum."""
        from scipy.ndimage import maximum_filter1d
        return maximum_filter1d(data, size=window, axis=0, mode='nearest')

    def _rolling_sum(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling sum."""
        return uniform_filter1d(data, size=window, axis=0, mode='nearest') * window

    def _compute_motion_energy(self, features: np.ndarray) -> np.ndarray:
        """Compute motion energy (sum of squared velocities)."""
        # Assume features include velocity components
        # Use variance as proxy for motion energy
        diff = np.diff(features, axis=0, prepend=features[:1])
        return np.sum(diff ** 2, axis=-1)

    def _compute_rate_of_change(self, features: np.ndarray) -> np.ndarray:
        """Compute first derivative of features."""
        rate = np.zeros_like(features)
        rate[1:-1] = (features[2:] - features[:-2]) / 2
        rate[0] = features[1] - features[0]
        rate[-1] = features[-1] - features[-2]
        return rate

    def _detect_change_points(
        self,
        features: np.ndarray,
        sensitivity: float = 2.0
    ) -> np.ndarray:
        """
        Detect potential behavior change points.
        Uses deviation from local trend as indicator.
        """
        n_frames = features.shape[0]

        # Smooth features
        smoothed = gaussian_filter1d(features, sigma=5, axis=0)

        # Compute deviation from trend
        deviation = np.abs(features - smoothed)
        deviation_score = np.mean(deviation, axis=-1)

        # Normalize by local std
        local_std = self._rolling_std(deviation_score[:, np.newaxis], 30)[:, 0]
        normalized_score = deviation_score / (local_std + 1e-8)

        # Threshold to get change point probability
        change_prob = 1 / (1 + np.exp(-sensitivity * (normalized_score - 1)))

        return change_prob


class SequenceAugmenter:
    """
    Data augmentation for temporal sequences.
    """

    def __init__(
        self,
        temporal_jitter_range: int = 5,
        noise_std: float = 0.01,
        dropout_prob: float = 0.1
    ):
        self.temporal_jitter_range = temporal_jitter_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def augment(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to feature sequence.

        Args:
            features: Shape (n_frames, n_features)
            labels: Shape (n_frames, n_classes) or (n_frames,)

        Returns:
            Augmented features and labels
        """
        features = features.copy()
        if labels is not None:
            labels = labels.copy()

        # Temporal jitter (shift sequence)
        if self.temporal_jitter_range > 0 and np.random.random() < 0.3:
            shift = np.random.randint(
                -self.temporal_jitter_range,
                self.temporal_jitter_range + 1
            )
            features = np.roll(features, shift, axis=0)
            if labels is not None:
                labels = np.roll(labels, shift, axis=0)

        # Add Gaussian noise
        if self.noise_std > 0 and np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, features.shape)
            features = features + noise.astype(features.dtype)

        # Random feature dropout
        if self.dropout_prob > 0 and np.random.random() < 0.3:
            mask = np.random.random(features.shape) > self.dropout_prob
            features = features * mask

        return features, labels

    def temporal_mixup(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation between two sequences.
        """
        lam = np.random.beta(alpha, alpha)

        mixed_features = lam * features1 + (1 - lam) * features2
        mixed_labels = lam * labels1 + (1 - lam) * labels2

        return mixed_features, mixed_labels

    def speed_augment(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        speed_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment by changing playback speed (temporal scaling).
        """
        from scipy.interpolate import interp1d

        n_frames = features.shape[0]
        speed = np.random.uniform(*speed_range)
        new_n_frames = int(n_frames / speed)

        # Create interpolation functions
        old_times = np.linspace(0, 1, n_frames)
        new_times = np.linspace(0, 1, new_n_frames)

        # Interpolate features
        f_features = interp1d(old_times, features, axis=0, kind='linear')
        new_features = f_features(new_times)

        # Interpolate labels
        if labels.ndim == 1:
            f_labels = interp1d(old_times, labels, kind='nearest')
        else:
            f_labels = interp1d(old_times, labels, axis=0, kind='nearest')
        new_labels = f_labels(new_times)

        # Resize back to original length by cropping or padding
        if new_n_frames > n_frames:
            # Random crop
            start = np.random.randint(0, new_n_frames - n_frames + 1)
            new_features = new_features[start:start + n_frames]
            new_labels = new_labels[start:start + n_frames]
        elif new_n_frames < n_frames:
            # Pad
            pad_len = n_frames - new_n_frames
            pad_before = pad_len // 2
            pad_after = pad_len - pad_before
            new_features = np.pad(
                new_features,
                [(pad_before, pad_after)] + [(0, 0)] * (new_features.ndim - 1),
                mode='edge'
            )
            if labels.ndim == 1:
                new_labels = np.pad(new_labels, (pad_before, pad_after), mode='edge')
            else:
                new_labels = np.pad(
                    new_labels,
                    [(pad_before, pad_after)] + [(0, 0)] * (new_labels.ndim - 1),
                    mode='edge'
                )

        return new_features.astype(features.dtype), new_labels.astype(labels.dtype)
