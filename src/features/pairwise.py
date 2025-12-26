"""
Pairwise interaction feature extraction for MABe behavior recognition.
Extracts features describing the spatial and dynamic relationship between two mice.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class PairwiseFeatureExtractor:
    """
    Extract pairwise interaction features between agent and target mice.

    Features extracted:
    - Relative position: distance, angle between mice
    - Relative velocity: approach/retreat speed, relative heading
    - Spatial relations: nose-to-body distances, facing angles
    - Contact proxies: minimum body part distance, overlap metrics
    """

    DEFAULT_PART_ORDER = [
        'nose', 'neck', 'body_center', 'tail_base',
        'ear_left', 'ear_right', 'lateral_left', 'lateral_right',
        'hip_left', 'hip_right', 'tail_tip'
    ]

    # Body parts to use for contact detection
    CONTACT_PARTS = ['nose', 'body_center', 'tail_base']

    def __init__(
        self,
        body_parts: Optional[List[str]] = None,
        fps: float = 30.0,
        smoothing_window: int = 3
    ):
        """
        Args:
            body_parts: List of body part names in order
            fps: Frame rate for velocity calculation
            smoothing_window: Window for smoothing derivatives
        """
        self.body_parts = body_parts or self.DEFAULT_PART_ORDER
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.dt = 1.0 / fps

        self.part_idx = {part: i for i, part in enumerate(self.body_parts)}

    def extract_features(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray,
        available_parts: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract pairwise features between agent and target.

        Args:
            agent_coords: Agent keypoints, shape (n_frames, n_bodyparts, 2)
            target_coords: Target keypoints, shape (n_frames, n_bodyparts, 2)
            available_parts: List of available body part names

        Returns:
            Dictionary of feature arrays
        """
        n_frames = agent_coords.shape[0]
        features = {}

        # Compute centroids
        agent_centroid = np.nanmean(agent_coords, axis=1)
        target_centroid = np.nanmean(target_coords, axis=1)

        # Relative position features
        rel_pos = target_centroid - agent_centroid
        distance = np.linalg.norm(rel_pos, axis=-1)
        features['distance'] = distance
        features['relative_x'] = rel_pos[:, 0]
        features['relative_y'] = rel_pos[:, 1]

        # Angle to target (in agent's reference frame)
        angle_to_target = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        features['angle_to_target'] = angle_to_target
        features['angle_to_target_sin'] = np.sin(angle_to_target)
        features['angle_to_target_cos'] = np.cos(angle_to_target)

        # Relative velocity features
        agent_velocity = self._compute_velocity(agent_centroid)
        target_velocity = self._compute_velocity(target_centroid)

        rel_velocity = target_velocity - agent_velocity
        features['relative_speed'] = np.linalg.norm(rel_velocity, axis=-1)
        features['relative_velocity_x'] = rel_velocity[:, 0]
        features['relative_velocity_y'] = rel_velocity[:, 1]

        # Approach/retreat speed (radial velocity component)
        rel_pos_unit = rel_pos / (distance[:, np.newaxis] + 1e-8)
        approach_speed = np.sum(rel_velocity * rel_pos_unit, axis=-1)
        features['approach_speed'] = approach_speed

        # Tangential velocity component
        # Use np.maximum to handle floating point precision issues where the value can be slightly negative
        tangent_speed_sq = np.sum(rel_velocity ** 2, axis=-1) - approach_speed ** 2
        tangent_speed = np.sqrt(np.maximum(0, tangent_speed_sq))
        features['tangent_speed'] = tangent_speed

        # Body orientation-based features
        if self._has_part('nose') and self._has_part('tail_base'):
            # Agent body orientation
            agent_orientation = self._compute_body_orientation(agent_coords)
            target_orientation = self._compute_body_orientation(target_coords)

            features['agent_orientation'] = agent_orientation
            features['target_orientation'] = target_orientation

            # Facing angle (is agent facing target?)
            facing_angle = self._angle_diff(agent_orientation, angle_to_target)
            features['facing_angle'] = facing_angle
            features['facing_angle_sin'] = np.sin(facing_angle)
            features['facing_angle_cos'] = np.cos(facing_angle)
            features['is_facing'] = (np.abs(facing_angle) < np.pi / 4).astype(np.float32)

            # Relative orientation (are mice facing same direction?)
            rel_orientation = self._angle_diff(agent_orientation, target_orientation)
            features['relative_orientation'] = rel_orientation
            features['relative_orientation_sin'] = np.sin(rel_orientation)
            features['relative_orientation_cos'] = np.cos(rel_orientation)

            # Target facing angle (is target facing agent?)
            angle_from_target = np.arctan2(-rel_pos[:, 1], -rel_pos[:, 0])
            target_facing = self._angle_diff(target_orientation, angle_from_target)
            features['target_facing_angle'] = target_facing
            features['mutual_facing'] = (
                (np.abs(facing_angle) < np.pi / 4) &
                (np.abs(target_facing) < np.pi / 4)
            ).astype(np.float32)

        # Nose-to-body distances
        if self._has_part('nose'):
            nose_distances = self._compute_nose_to_body_distances(
                agent_coords, target_coords
            )
            features.update(nose_distances)

        # Minimum body part distance (contact proxy)
        min_dist = self._compute_minimum_distance(agent_coords, target_coords)
        features['min_body_distance'] = min_dist
        features['is_close'] = (min_dist < 0.1).astype(np.float32)  # Threshold in normalized units

        # Specific body part distances
        for part in ['nose', 'body_center', 'tail_base']:
            if self._has_part(part):
                part_dist = self._compute_part_to_part_distance(
                    agent_coords, target_coords, part, part
                )
                features[f'{part}_to_{part}_distance'] = part_dist

        # Area overlap (bounding box intersection)
        overlap = self._compute_bbox_overlap(agent_coords, target_coords)
        features['bbox_overlap'] = overlap
        features['bbox_overlap_ratio'] = overlap / (
            self._compute_bbox_area(agent_coords) + 1e-8
        )

        return features

    def extract_feature_vector(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray,
        available_parts: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract features as a single concatenated array.

        Returns:
            Feature array of shape (n_frames, n_features)
        """
        features = self.extract_features(agent_coords, target_coords, available_parts)
        return np.stack(list(features.values()), axis=-1)

    def _has_part(self, part_name: str) -> bool:
        """Check if body part is available."""
        return part_name in self.part_idx

    def _get_part_coords(self, coords: np.ndarray, part_name: str) -> np.ndarray:
        """Get coordinates for a specific body part."""
        if part_name not in self.part_idx:
            raise ValueError(f"Body part '{part_name}' not found")
        return coords[:, self.part_idx[part_name], :]

    def _compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Compute velocity from positions."""
        velocity = np.zeros_like(positions)
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.dt)
        velocity[0] = (positions[1] - positions[0]) / self.dt
        velocity[-1] = (positions[-1] - positions[-2]) / self.dt

        if self.smoothing_window > 1:
            from scipy.ndimage import uniform_filter1d
            velocity = uniform_filter1d(velocity, size=self.smoothing_window, axis=0)

        return velocity

    def _compute_body_orientation(self, coords: np.ndarray) -> np.ndarray:
        """Compute body orientation from nose to tail."""
        nose = self._get_part_coords(coords, 'nose')
        tail = self._get_part_coords(coords, 'tail_base')
        direction = nose - tail
        return np.arctan2(direction[:, 1], direction[:, 0])

    def _angle_diff(self, angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        """Compute angle difference with proper wrapping."""
        diff = angle1 - angle2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def _compute_nose_to_body_distances(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute distance from agent's nose to target's body parts."""
        features = {}
        agent_nose = self._get_part_coords(agent_coords, 'nose')

        # Distance to target's specific parts
        for part in ['nose', 'body_center', 'tail_base']:
            if self._has_part(part):
                target_part = self._get_part_coords(target_coords, part)
                dist = np.linalg.norm(agent_nose - target_part, axis=-1)
                features[f'nose_to_target_{part}'] = dist

        # Distance to target's centroid
        target_centroid = np.nanmean(target_coords, axis=1)
        features['nose_to_target_centroid'] = np.linalg.norm(
            agent_nose - target_centroid, axis=-1
        )

        return features

    def _compute_minimum_distance(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray
    ) -> np.ndarray:
        """Compute minimum distance between any body parts."""
        n_frames = agent_coords.shape[0]
        n_parts = agent_coords.shape[1]

        min_dist = np.full(n_frames, np.inf)

        for i in range(n_parts):
            for j in range(n_parts):
                dist = np.linalg.norm(
                    agent_coords[:, i, :] - target_coords[:, j, :],
                    axis=-1
                )
                min_dist = np.minimum(min_dist, dist)

        return min_dist

    def _compute_part_to_part_distance(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray,
        agent_part: str,
        target_part: str
    ) -> np.ndarray:
        """Compute distance between specific body parts."""
        agent_p = self._get_part_coords(agent_coords, agent_part)
        target_p = self._get_part_coords(target_coords, target_part)
        return np.linalg.norm(agent_p - target_p, axis=-1)

    def _compute_bbox_overlap(
        self,
        agent_coords: np.ndarray,
        target_coords: np.ndarray
    ) -> np.ndarray:
        """Compute bounding box overlap area."""
        # Agent bbox
        agent_min = np.nanmin(agent_coords, axis=1)
        agent_max = np.nanmax(agent_coords, axis=1)

        # Target bbox
        target_min = np.nanmin(target_coords, axis=1)
        target_max = np.nanmax(target_coords, axis=1)

        # Intersection
        inter_min = np.maximum(agent_min, target_min)
        inter_max = np.minimum(agent_max, target_max)

        inter_size = np.maximum(0, inter_max - inter_min)
        overlap_area = inter_size[:, 0] * inter_size[:, 1]

        return overlap_area

    def _compute_bbox_area(self, coords: np.ndarray) -> np.ndarray:
        """Compute bounding box area."""
        min_coords = np.nanmin(coords, axis=1)
        max_coords = np.nanmax(coords, axis=1)
        size = max_coords - min_coords
        return size[:, 0] * size[:, 1]

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'distance', 'relative_x', 'relative_y',
            'angle_to_target', 'angle_to_target_sin', 'angle_to_target_cos',
            'relative_speed', 'relative_velocity_x', 'relative_velocity_y',
            'approach_speed', 'tangent_speed',
            'agent_orientation', 'target_orientation',
            'facing_angle', 'facing_angle_sin', 'facing_angle_cos', 'is_facing',
            'relative_orientation', 'relative_orientation_sin', 'relative_orientation_cos',
            'target_facing_angle', 'mutual_facing',
            'nose_to_target_nose', 'nose_to_target_body_center', 'nose_to_target_tail_base',
            'nose_to_target_centroid',
            'min_body_distance', 'is_close',
            'nose_to_nose_distance', 'body_center_to_body_center_distance',
            'tail_base_to_tail_base_distance',
            'bbox_overlap', 'bbox_overlap_ratio'
        ]
