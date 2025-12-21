"""
Single mouse feature extraction for MABe behavior recognition.
Extracts position, velocity, pose, and posture features for individual mice.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SingleMouseFeatureExtractor:
    """
    Extract behavioral features from single mouse keypoint trajectories.

    Features extracted:
    - Position: centroid, bounding box
    - Velocity: speed, heading direction, angular velocity
    - Acceleration: linear and angular
    - Pose: body orientation, body length/width
    - Posture: nose-tail angle, body curvature
    """

    # Default body part indices (can be overridden)
    DEFAULT_PART_ORDER = [
        'nose', 'neck', 'body_center', 'tail_base',
        'ear_left', 'ear_right', 'lateral_left', 'lateral_right',
        'hip_left', 'hip_right', 'tail_tip'
    ]

    def __init__(
        self,
        body_parts: Optional[List[str]] = None,
        fps: float = 30.0,
        smoothing_window: int = 3
    ):
        """
        Args:
            body_parts: List of body part names in order
            fps: Frame rate for velocity/acceleration calculation
            smoothing_window: Window size for velocity smoothing
        """
        self.body_parts = body_parts or self.DEFAULT_PART_ORDER
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.dt = 1.0 / fps

        # Build part index
        self.part_idx = {part: i for i, part in enumerate(self.body_parts)}

    def extract_features(
        self,
        coords: np.ndarray,
        available_parts: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract all features from mouse coordinates.

        Args:
            coords: Keypoint coordinates, shape (n_frames, n_bodyparts, 2)
            available_parts: List of available body part names

        Returns:
            Dictionary of feature arrays
        """
        n_frames = coords.shape[0]
        features = {}

        # Position features
        centroid = self._compute_centroid(coords)
        features['centroid_x'] = centroid[:, 0]
        features['centroid_y'] = centroid[:, 1]

        bbox = self._compute_bounding_box(coords)
        features['bbox_width'] = bbox[:, 0]
        features['bbox_height'] = bbox[:, 1]
        features['bbox_area'] = bbox[:, 0] * bbox[:, 1]

        # Velocity features
        velocity = self._compute_velocity(centroid)
        speed = np.linalg.norm(velocity, axis=-1)
        features['speed'] = speed
        features['velocity_x'] = velocity[:, 0]
        features['velocity_y'] = velocity[:, 1]

        heading = np.arctan2(velocity[:, 1], velocity[:, 0])
        features['heading'] = heading
        features['heading_sin'] = np.sin(heading)
        features['heading_cos'] = np.cos(heading)

        angular_velocity = self._compute_angular_velocity(heading)
        features['angular_velocity'] = angular_velocity

        # Acceleration features
        acceleration = self._compute_acceleration(velocity)
        features['acceleration'] = np.linalg.norm(acceleration, axis=-1)
        features['acceleration_x'] = acceleration[:, 0]
        features['acceleration_y'] = acceleration[:, 1]

        angular_acc = self._compute_angular_velocity(angular_velocity)
        features['angular_acceleration'] = angular_acc

        # Pose features (if body parts available)
        if self._has_part('nose') and self._has_part('tail_base'):
            orientation = self._compute_body_orientation(coords)
            features['body_orientation'] = orientation
            features['body_orientation_sin'] = np.sin(orientation)
            features['body_orientation_cos'] = np.cos(orientation)

            body_length = self._compute_body_length(coords)
            features['body_length'] = body_length

        if self._has_part('lateral_left') and self._has_part('lateral_right'):
            body_width = self._compute_body_width(coords)
            features['body_width'] = body_width

        # Posture features
        if self._has_part('nose') and self._has_part('body_center') and self._has_part('tail_base'):
            curvature = self._compute_body_curvature(coords)
            features['body_curvature'] = curvature

            nose_tail_angle = self._compute_nose_tail_angle(coords)
            features['nose_tail_angle'] = nose_tail_angle

        # Motion energy
        features['motion_energy'] = speed ** 2

        return features

    def extract_feature_vector(
        self,
        coords: np.ndarray,
        available_parts: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract features as a single concatenated array.

        Returns:
            Feature array of shape (n_frames, n_features)
        """
        features = self.extract_features(coords, available_parts)
        return np.stack(list(features.values()), axis=-1)

    def _has_part(self, part_name: str) -> bool:
        """Check if body part is available."""
        return part_name in self.part_idx

    def _get_part_coords(self, coords: np.ndarray, part_name: str) -> np.ndarray:
        """Get coordinates for a specific body part."""
        if part_name not in self.part_idx:
            raise ValueError(f"Body part '{part_name}' not found")
        return coords[:, self.part_idx[part_name], :]

    def _compute_centroid(self, coords: np.ndarray) -> np.ndarray:
        """Compute centroid (mean of all body parts)."""
        # Handle NaN values
        valid_mask = ~np.isnan(coords).any(axis=-1)
        centroid = np.nanmean(coords, axis=1)
        return centroid

    def _compute_bounding_box(self, coords: np.ndarray) -> np.ndarray:
        """Compute bounding box dimensions (width, height)."""
        min_coords = np.nanmin(coords, axis=1)
        max_coords = np.nanmax(coords, axis=1)
        bbox = max_coords - min_coords
        return bbox

    def _compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Compute velocity from positions using central difference."""
        n_frames = positions.shape[0]
        velocity = np.zeros_like(positions)

        # Central difference for interior points
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.dt)

        # Forward/backward difference for endpoints
        velocity[0] = (positions[1] - positions[0]) / self.dt
        velocity[-1] = (positions[-1] - positions[-2]) / self.dt

        # Smooth velocity
        if self.smoothing_window > 1:
            from scipy.ndimage import uniform_filter1d
            velocity = uniform_filter1d(velocity, size=self.smoothing_window, axis=0)

        return velocity

    def _compute_angular_velocity(self, angles: np.ndarray) -> np.ndarray:
        """Compute angular velocity with proper angle wrapping."""
        n_frames = len(angles)
        angular_vel = np.zeros(n_frames)

        # Compute angle differences with wrapping
        angle_diff = np.diff(angles)
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        angular_vel[1:] = angle_diff / self.dt

        # Smooth
        if self.smoothing_window > 1:
            from scipy.ndimage import uniform_filter1d
            angular_vel = uniform_filter1d(angular_vel, size=self.smoothing_window)

        return angular_vel

    def _compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute acceleration from velocity."""
        return self._compute_velocity(velocity)

    def _compute_body_orientation(self, coords: np.ndarray) -> np.ndarray:
        """Compute body orientation (angle from tail to nose)."""
        nose = self._get_part_coords(coords, 'nose')
        tail = self._get_part_coords(coords, 'tail_base')

        direction = nose - tail
        orientation = np.arctan2(direction[:, 1], direction[:, 0])

        return orientation

    def _compute_body_length(self, coords: np.ndarray) -> np.ndarray:
        """Compute body length (nose to tail distance)."""
        nose = self._get_part_coords(coords, 'nose')
        tail = self._get_part_coords(coords, 'tail_base')

        return np.linalg.norm(nose - tail, axis=-1)

    def _compute_body_width(self, coords: np.ndarray) -> np.ndarray:
        """Compute body width (lateral left to right distance)."""
        left = self._get_part_coords(coords, 'lateral_left')
        right = self._get_part_coords(coords, 'lateral_right')

        return np.linalg.norm(left - right, axis=-1)

    def _compute_body_curvature(self, coords: np.ndarray) -> np.ndarray:
        """Compute body curvature (deviation from straight line)."""
        nose = self._get_part_coords(coords, 'nose')
        center = self._get_part_coords(coords, 'body_center')
        tail = self._get_part_coords(coords, 'tail_base')

        # Compute perpendicular distance of center from nose-tail line
        line_vec = tail - nose
        line_len = np.linalg.norm(line_vec, axis=-1, keepdims=True)
        line_unit = line_vec / (line_len + 1e-8)

        # Vector from nose to center
        nose_to_center = center - nose

        # Project onto line
        proj_len = np.sum(nose_to_center * line_unit, axis=-1, keepdims=True)
        proj = proj_len * line_unit

        # Perpendicular distance
        perp = nose_to_center - proj
        curvature = np.linalg.norm(perp, axis=-1)

        # Sign based on cross product
        cross = line_vec[:, 0] * nose_to_center[:, 1] - line_vec[:, 1] * nose_to_center[:, 0]
        curvature = curvature * np.sign(cross)

        return curvature

    def _compute_nose_tail_angle(self, coords: np.ndarray) -> np.ndarray:
        """Compute angle between nose-center and center-tail vectors."""
        nose = self._get_part_coords(coords, 'nose')
        center = self._get_part_coords(coords, 'body_center')
        tail = self._get_part_coords(coords, 'tail_base')

        vec1 = nose - center
        vec2 = tail - center

        # Compute angle between vectors
        dot = np.sum(vec1 * vec2, axis=-1)
        cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]

        angle = np.arctan2(cross, dot)

        return angle

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        # This is a simplified version; actual names depend on available parts
        return [
            'centroid_x', 'centroid_y',
            'bbox_width', 'bbox_height', 'bbox_area',
            'speed', 'velocity_x', 'velocity_y',
            'heading', 'heading_sin', 'heading_cos',
            'angular_velocity',
            'acceleration', 'acceleration_x', 'acceleration_y',
            'angular_acceleration',
            'body_orientation', 'body_orientation_sin', 'body_orientation_cos',
            'body_length', 'body_width',
            'body_curvature', 'nose_tail_angle',
            'motion_energy'
        ]
