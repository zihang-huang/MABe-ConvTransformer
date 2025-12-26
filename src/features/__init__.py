"""
Engineered feature extraction for MABe behavior recognition.
"""

from .single_mouse import SingleMouseFeatureExtractor
from .pairwise import PairwiseFeatureExtractor
from .temporal import TemporalFeatureExtractor, SequenceAugmenter

import numpy as np
from typing import Dict, List, Optional, Tuple


class CombinedFeatureExtractor:
    """
    Combined feature extractor that integrates single mouse, pairwise,
    and temporal features for behavior recognition.

    This class orchestrates the extraction of:
    - Raw keypoint coordinates (baseline)
    - Single mouse features (velocity, acceleration, orientation, pose)
    - Pairwise interaction features (distance, facing, relative motion)
    - Temporal context features (rolling statistics, motion energy)
    """

    def __init__(
        self,
        body_parts: Optional[List[str]] = None,
        fps: float = 30.0,
        # Feature toggles
        use_raw_coords: bool = True,
        use_single_mouse_features: bool = True,
        use_pairwise_features: bool = True,
        use_temporal_features: bool = True,
        # Temporal settings
        temporal_windows: Optional[List[int]] = None,
        # Feature subset selection
        single_mouse_config: Optional[Dict] = None,
        pairwise_config: Optional[Dict] = None
    ):
        """
        Args:
            body_parts: List of body part names in order
            fps: Frame rate for velocity/acceleration calculation
            use_raw_coords: Include raw keypoint coordinates
            use_single_mouse_features: Include velocity, acceleration, etc.
            use_pairwise_features: Include distance, facing angle, etc.
            use_temporal_features: Include rolling statistics
            temporal_windows: Window sizes for temporal features (default: [5, 15, 30, 60])
            single_mouse_config: Config dict for single mouse feature selection
            pairwise_config: Config dict for pairwise feature selection
        """
        self.body_parts = body_parts or [
            'nose', 'neck', 'body_center', 'tail_base',
            'ear_left', 'ear_right', 'lateral_left', 'lateral_right',
            'hip_left', 'hip_right', 'tail_tip'
        ]
        self.fps = fps

        # Feature toggles
        self.use_raw_coords = use_raw_coords
        self.use_single_mouse_features = use_single_mouse_features
        self.use_pairwise_features = use_pairwise_features
        self.use_temporal_features = use_temporal_features

        # Config for feature subsets
        self.single_mouse_config = single_mouse_config or {}
        self.pairwise_config = pairwise_config or {}

        # Initialize extractors
        if use_single_mouse_features:
            self.single_extractor = SingleMouseFeatureExtractor(
                body_parts=self.body_parts,
                fps=fps
            )

        if use_pairwise_features:
            self.pairwise_extractor = PairwiseFeatureExtractor(
                body_parts=self.body_parts,
                fps=fps
            )

        if use_temporal_features:
            self.temporal_extractor = TemporalFeatureExtractor(
                window_sizes=temporal_windows or [5, 15, 30, 60],
                fps=fps
            )

        # Cache for feature dimension
        self._feature_dim: Optional[int] = None
        self._feature_names: Optional[List[str]] = None

    def extract_features(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray,
        include_temporal: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all configured features from agent and target coordinates.

        Args:
            agent_coords: Agent keypoint coordinates, shape (n_frames, n_bodyparts, 2)
            target_coords: Target keypoint coordinates, shape (n_frames, n_bodyparts, 2)
            include_temporal: Whether to include temporal features (can be disabled for speed)

        Returns:
            Tuple of (feature_array, feature_names)
            - feature_array: Shape (n_frames, n_features)
            - feature_names: List of feature names
        """
        n_frames = agent_coords.shape[0]
        all_features = []
        all_names = []

        # 1. Raw keypoint coordinates (baseline features)
        if self.use_raw_coords:
            agent_flat = agent_coords.reshape(n_frames, -1)
            target_flat = target_coords.reshape(n_frames, -1)

            all_features.append(agent_flat)
            all_features.append(target_flat)

            for mouse in ['agent', 'target']:
                for bp in self.body_parts:
                    all_names.extend([f'{mouse}_{bp}_x', f'{mouse}_{bp}_y'])

        # 2. Single mouse features
        if self.use_single_mouse_features:
            agent_single = self._extract_single_mouse(agent_coords, prefix='agent')
            target_single = self._extract_single_mouse(target_coords, prefix='target')

            all_features.append(agent_single[0])
            all_features.append(target_single[0])
            all_names.extend(agent_single[1])
            all_names.extend(target_single[1])

        # 3. Pairwise interaction features
        if self.use_pairwise_features:
            pairwise = self._extract_pairwise(agent_coords, target_coords)
            all_features.append(pairwise[0])
            all_names.extend(pairwise[1])

        # Concatenate base features
        base_features = np.concatenate(all_features, axis=-1)
        base_names = all_names.copy()

        # 4. Temporal features (computed on top of base features)
        if self.use_temporal_features and include_temporal:
            # Select a subset of base features for temporal analysis
            # to avoid feature explosion
            temporal_input_features, temporal_input_names = self._select_temporal_input(
                base_features, base_names
            )

            temporal = self.temporal_extractor.extract_features(
                temporal_input_features,
                temporal_input_names
            )

            temporal_features = np.stack(list(temporal.values()), axis=-1)
            temporal_names = list(temporal.keys())

            final_features = np.concatenate([base_features, temporal_features], axis=-1)
            final_names = base_names + temporal_names
        else:
            final_features = base_features
            final_names = base_names

        return final_features.astype(np.float32), final_names

    def _extract_single_mouse(
        self,
        coords: np.ndarray,
        prefix: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract single mouse features with optional filtering."""
        features_dict = self.single_extractor.extract_features(coords)

        # Apply feature selection from config
        selected_features = []
        selected_names = []

        config = self.single_mouse_config

        for name, values in features_dict.items():
            include = True

            # Filter based on config
            if 'position' in name or 'centroid' in name or 'bbox' in name:
                include = config.get('position', True)
            elif 'velocity' in name or 'speed' in name or 'heading' in name:
                include = config.get('velocity', True)
            elif 'acceleration' in name:
                include = config.get('acceleration', True)
            elif 'orientation' in name or 'body_orientation' in name:
                include = config.get('orientation', True)
            elif 'body_length' in name or 'body_width' in name or 'curvature' in name:
                include = config.get('body_shape', True)

            if include:
                selected_features.append(values)
                selected_names.append(f'{prefix}_{name}')

        if not selected_features:
            return np.zeros((coords.shape[0], 0), dtype=np.float32), []

        return np.stack(selected_features, axis=-1), selected_names

    def _extract_pairwise(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract pairwise features with optional filtering."""
        features_dict = self.pairwise_extractor.extract_features(
            agent_coords, target_coords
        )

        # Apply feature selection from config
        selected_features = []
        selected_names = []

        config = self.pairwise_config

        for name, values in features_dict.items():
            include = True

            # Filter based on config
            if 'distance' in name and 'velocity' not in name:
                include = config.get('distance', True)
            elif 'relative_x' in name or 'relative_y' in name:
                include = config.get('relative_position', True)
            elif 'velocity' in name or 'speed' in name:
                include = config.get('relative_velocity', True)
            elif 'facing' in name or 'orientation' in name:
                include = config.get('facing_angle', True)
            elif 'angle_to_target' in name:
                include = config.get('relative_position', True)

            if include:
                selected_features.append(values)
                selected_names.append(f'pair_{name}')

        if not selected_features:
            return np.zeros((agent_coords.shape[0], 0), dtype=np.float32), []

        return np.stack(selected_features, axis=-1), selected_names

    def _select_temporal_input(
        self,
        features: np.ndarray,
        names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select a subset of features for temporal analysis to avoid feature explosion.

        Prioritizes motion-related features that benefit from temporal context:
        - Speed/velocity features
        - Distance features
        - Acceleration features
        """
        priority_patterns = [
            'speed', 'velocity', 'distance', 'acceleration',
            'approach', 'facing', 'motion_energy'
        ]

        selected_indices = []
        selected_names = []

        for i, name in enumerate(names):
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in priority_patterns):
                selected_indices.append(i)
                selected_names.append(name)

        # Limit to avoid too many temporal features
        max_temporal_inputs = 20
        if len(selected_indices) > max_temporal_inputs:
            selected_indices = selected_indices[:max_temporal_inputs]
            selected_names = selected_names[:max_temporal_inputs]

        if not selected_indices:
            # Fallback: use first few features
            n_use = min(10, features.shape[-1])
            return features[:, :n_use], names[:n_use]

        return features[:, selected_indices], selected_names

    def get_feature_dim(
        self,
        n_bodyparts: Optional[int] = None,
        include_temporal: bool = True
    ) -> int:
        """
        Estimate the output feature dimension.

        This is an approximation - actual dim depends on available body parts.
        """
        if n_bodyparts is None:
            n_bodyparts = len(self.body_parts)

        dim = 0

        # Raw coordinates: 2 mice * n_bodyparts * 2 (x, y)
        if self.use_raw_coords:
            dim += 2 * n_bodyparts * 2

        # Single mouse features: ~24 features per mouse
        if self.use_single_mouse_features:
            dim += 2 * 24

        # Pairwise features: ~30 features
        if self.use_pairwise_features:
            dim += 33

        # Temporal features: varies based on windows and input features
        if self.use_temporal_features and include_temporal:
            # Roughly: 20 input features * 4 windows * 5 stats + motion energy + change points
            dim += 420  # Approximate

        return dim


__all__ = [
    'SingleMouseFeatureExtractor',
    'PairwiseFeatureExtractor',
    'TemporalFeatureExtractor',
    'SequenceAugmenter',
    'CombinedFeatureExtractor'
]
