"""
Data preprocessing utilities for MABe behavior recognition.
Handles coordinate normalization, temporal resampling, and missing data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import interpolate
from scipy.ndimage import uniform_filter1d


class CoordinateNormalizer:
    """
    Normalizes keypoint coordinates using video/arena metadata.

    Transformations:
    1. Convert pixels to centimeters using pix_per_cm
    2. Normalize to arena dimensions (0-1 range)
    3. Center coordinates to arena center
    """

    def __init__(
        self,
        normalize_to_arena: bool = True,
        center_coordinates: bool = True,
        use_cm: bool = True
    ):
        self.normalize_to_arena = normalize_to_arena
        self.center_coordinates = center_coordinates
        self.use_cm = use_cm

    def __call__(
        self,
        coords: np.ndarray,
        metadata: Dict
    ) -> np.ndarray:
        """
        Normalize coordinates.

        Args:
            coords: Array of shape (n_frames, n_mice, n_bodyparts, 2) or (n_frames, n_bodyparts, 2)
            metadata: Video metadata dict with pix_per_cm, arena dimensions, etc.

        Returns:
            Normalized coordinates with same shape as input
        """
        coords = coords.copy().astype(np.float32)

        # Get metadata values with defaults
        pix_per_cm = metadata.get('pix_per_cm_approx', metadata.get('pix per cm (approx)', 1.0))
        arena_width = metadata.get('arena_width_cm', metadata.get('arena width (cm)', None))
        arena_height = metadata.get('arena_height_cm', metadata.get('arena height (cm)', None))
        video_width = metadata.get('video_width', metadata.get('video width', None))
        video_height = metadata.get('video_height', metadata.get('video height', None))

        # Handle NaN/None values
        if pix_per_cm is None or (isinstance(pix_per_cm, float) and np.isnan(pix_per_cm)):
            pix_per_cm = 1.0

        # Convert to centimeters
        if self.use_cm and pix_per_cm > 0:
            coords = coords / pix_per_cm

        # Normalize to arena dimensions
        if self.normalize_to_arena:
            if arena_width is not None and arena_height is not None:
                if not np.isnan(arena_width) and not np.isnan(arena_height):
                    coords[..., 0] = coords[..., 0] / arena_width
                    coords[..., 1] = coords[..., 1] / arena_height
            elif video_width is not None and video_height is not None:
                # Fall back to video dimensions
                if pix_per_cm > 0:
                    video_width_cm = video_width / pix_per_cm
                    video_height_cm = video_height / pix_per_cm
                    coords[..., 0] = coords[..., 0] / video_width_cm
                    coords[..., 1] = coords[..., 1] / video_height_cm

        # Center coordinates (shift to [-0.5, 0.5] range)
        if self.center_coordinates:
            coords = coords - 0.5

        return coords


class TemporalResampler:
    """
    Resample tracking data to a consistent frame rate.
    Uses linear interpolation for smooth transitions.
    """

    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps

    def __call__(
        self,
        data: np.ndarray,
        source_fps: float,
        axis: int = 0
    ) -> np.ndarray:
        """
        Resample temporal data to target FPS.

        Args:
            data: Array with time dimension along specified axis
            source_fps: Original frame rate
            axis: Time axis (default 0)

        Returns:
            Resampled array
        """
        if source_fps == self.target_fps:
            return data

        n_frames = data.shape[axis]
        duration = n_frames / source_fps
        n_target_frames = int(duration * self.target_fps)

        # Create interpolation function
        source_times = np.linspace(0, duration, n_frames)
        target_times = np.linspace(0, duration, n_target_frames)

        # Handle multi-dimensional data
        if data.ndim == 1:
            f = interpolate.interp1d(source_times, data, kind='linear', fill_value='extrapolate')
            return f(target_times)

        # Move time axis to first position for easier processing
        data = np.moveaxis(data, axis, 0)
        original_shape = data.shape

        # Reshape to (n_frames, -1) for interpolation
        data_flat = data.reshape(n_frames, -1)

        # Interpolate each feature
        resampled = np.zeros((n_target_frames, data_flat.shape[1]), dtype=data.dtype)
        for i in range(data_flat.shape[1]):
            f = interpolate.interp1d(source_times, data_flat[:, i], kind='linear', fill_value='extrapolate')
            resampled[:, i] = f(target_times)

        # Reshape back and move axis
        new_shape = (n_target_frames,) + original_shape[1:]
        resampled = resampled.reshape(new_shape)
        resampled = np.moveaxis(resampled, 0, axis)

        return resampled


class MissingDataHandler:
    """
    Handle missing keypoint data through interpolation and masking.
    """

    def __init__(
        self,
        max_gap_frames: int = 5,
        confidence_threshold: float = 0.3
    ):
        self.max_gap_frames = max_gap_frames
        self.confidence_threshold = confidence_threshold

    def interpolate_missing(
        self,
        coords: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate missing keypoints within short gaps.

        Args:
            coords: Coordinates array (n_frames, n_bodyparts, 2) or similar
            confidence: Optional confidence scores for each keypoint

        Returns:
            Tuple of (interpolated_coords, valid_mask)
        """
        coords = coords.copy()

        # Detect missing data (NaN or zero coordinates)
        missing_mask = np.isnan(coords).any(axis=-1) | (coords == 0).all(axis=-1)

        # If confidence provided, also mark low-confidence as missing
        if confidence is not None:
            low_conf = confidence < self.confidence_threshold
            missing_mask = missing_mask | low_conf

        valid_mask = ~missing_mask

        # Interpolate short gaps for each bodypart
        n_frames = coords.shape[0]

        if coords.ndim == 3:  # (n_frames, n_bodyparts, 2)
            n_bodyparts = coords.shape[1]
            for bp in range(n_bodyparts):
                coords[:, bp, :] = self._interpolate_1d_gaps(
                    coords[:, bp, :],
                    missing_mask[:, bp]
                )
        elif coords.ndim == 2:  # (n_frames, 2)
            coords = self._interpolate_1d_gaps(coords, missing_mask)

        return coords, valid_mask

    def _interpolate_1d_gaps(
        self,
        data: np.ndarray,
        missing: np.ndarray
    ) -> np.ndarray:
        """Interpolate gaps shorter than max_gap_frames."""
        n_frames = len(data)

        # Find gap regions
        gap_starts = []
        gap_ends = []
        in_gap = False

        for i in range(n_frames):
            if missing[i] and not in_gap:
                gap_starts.append(i)
                in_gap = True
            elif not missing[i] and in_gap:
                gap_ends.append(i)
                in_gap = False

        if in_gap:
            gap_ends.append(n_frames)

        # Interpolate short gaps
        for start, end in zip(gap_starts, gap_ends):
            gap_length = end - start

            if gap_length <= self.max_gap_frames:
                # Get boundary values for interpolation
                left_idx = start - 1 if start > 0 else None
                right_idx = end if end < n_frames else None

                if left_idx is not None and right_idx is not None:
                    # Linear interpolation
                    for i in range(start, end):
                        alpha = (i - start + 1) / (gap_length + 1)
                        data[i] = (1 - alpha) * data[left_idx] + alpha * data[right_idx]
                elif left_idx is not None:
                    # Use left value
                    data[start:end] = data[left_idx]
                elif right_idx is not None:
                    # Use right value
                    data[start:end] = data[right_idx]

        return data

    def create_attention_mask(
        self,
        valid_mask: np.ndarray,
        window_size: int = 1
    ) -> np.ndarray:
        """
        Create attention mask for transformer/model input.
        Smooths the valid mask to handle edge cases.
        """
        if window_size > 1:
            # Smooth the mask
            smoothed = uniform_filter1d(valid_mask.astype(float), size=window_size, axis=0)
            return (smoothed > 0.5).astype(np.float32)
        return valid_mask.astype(np.float32)


class BodyPartMapper:
    """
    Handle different body part configurations across labs.
    Maps diverse body part sets to a common representation.
    """

    # Standard body part set
    STANDARD_PARTS = [
        'nose', 'neck', 'body_center', 'tail_base',
        'ear_left', 'ear_right',
        'lateral_left', 'lateral_right',
        'hip_left', 'hip_right',
        'tail_tip'
    ]

    # Core parts available in most tracking configs
    CORE_PARTS = ['nose', 'neck', 'body_center', 'tail_base']

    # Mappings for alternative naming conventions
    PART_ALIASES = {
        'snout': 'nose',
        'head': 'neck',
        'center': 'body_center',
        'centroid': 'body_center',
        'tailbase': 'tail_base',
        'tailtip': 'tail_tip',
        'left_ear': 'ear_left',
        'right_ear': 'ear_right',
        'left_hip': 'hip_left',
        'right_hip': 'hip_right',
    }

    def __init__(self, use_core_only: bool = False):
        self.use_core_only = use_core_only
        self.target_parts = self.CORE_PARTS if use_core_only else self.STANDARD_PARTS

    def map_bodyparts(
        self,
        coords: Dict[str, np.ndarray],
        available_parts: List[str]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Map available body parts to standard representation.

        Args:
            coords: Dict mapping body part names to coordinate arrays
            available_parts: List of body part names in the input

        Returns:
            Tuple of (mapped_coords, part_names, availability_mask)
        """
        n_frames = next(iter(coords.values())).shape[0]
        n_parts = len(self.target_parts)

        mapped_coords = np.full((n_frames, n_parts, 2), np.nan, dtype=np.float32)
        availability_mask = np.zeros(n_parts, dtype=bool)

        # Normalize part names
        normalized_available = {}
        for part in available_parts:
            normalized = part.lower().replace(' ', '_').replace('-', '_')
            normalized = self.PART_ALIASES.get(normalized, normalized)
            if part in coords:
                normalized_available[normalized] = coords[part]

        # Map to standard parts
        for i, target_part in enumerate(self.target_parts):
            if target_part in normalized_available:
                mapped_coords[:, i, :] = normalized_available[target_part]
                availability_mask[i] = True

        return mapped_coords, self.target_parts, availability_mask

    def compute_derived_parts(
        self,
        coords: np.ndarray,
        part_names: List[str],
        availability: np.ndarray
    ) -> np.ndarray:
        """
        Compute derived body parts from available ones.
        E.g., compute body_center from nose and tail_base if not available.
        """
        coords = coords.copy()
        part_idx = {name: i for i, name in enumerate(part_names)}

        # Compute body_center if missing but nose and tail_base available
        if 'body_center' in part_idx and not availability[part_idx['body_center']]:
            if ('nose' in part_idx and availability[part_idx['nose']] and
                'tail_base' in part_idx and availability[part_idx['tail_base']]):
                nose_coords = coords[:, part_idx['nose'], :]
                tail_coords = coords[:, part_idx['tail_base'], :]
                coords[:, part_idx['body_center'], :] = (nose_coords + tail_coords) / 2
                availability[part_idx['body_center']] = True

        # Compute neck if missing but nose and body_center available
        if 'neck' in part_idx and not availability[part_idx['neck']]:
            if ('nose' in part_idx and availability[part_idx['nose']] and
                'body_center' in part_idx and availability[part_idx['body_center']]):
                nose_coords = coords[:, part_idx['nose'], :]
                center_coords = coords[:, part_idx['body_center'], :]
                coords[:, part_idx['neck'], :] = 0.7 * nose_coords + 0.3 * center_coords
                availability[part_idx['neck']] = True

        return coords
