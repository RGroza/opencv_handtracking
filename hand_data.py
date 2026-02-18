import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Deque


@dataclass
class HandData:
    """Per-hand state container updated every frame."""

    side: str  # 'left' | 'right'
    smoothing_window: int = 20
    detected: bool = False

    # Smoothed pose tuple in camera normalized (0.0-1.0) coordinates:
    # (x_n, y_n, z_n, qw, qx, qy, qz, index, pinky, thumb)
    pose: Optional[tuple] = None

    # Raw (pre-smoothing) camera-normalized position (x_n, y_n, z_n) for this frame.
    raw_cam_pos: Optional[np.ndarray] = None

    # Smoothed axes in image space: (origin_px, x_axis, y_axis, z_axis)
    axes: Optional[Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray]] = None

    # Previous (last selected) relative rotation matrix used for temporal disambiguation.
    prev_rel_rot_mat: Optional[np.ndarray] = None

    # Finger values (index, pinky, thumb) in [0,1]
    finger_values: Optional[Tuple[float, float, float]] = None

    # Per-hand smoothing histories
    pose_history: Deque[tuple] = field(init=False)
    axes_history: Deque[Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray]] = field(init=False)

    # Per-hand episode offset (camera-normalized xyz) captured on START
    start_offset_cam: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # Per-hand robot output offset (meters)
    robot_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


    def __post_init__(self) -> None:
        self.pose_history = deque(maxlen=int(self.smoothing_window))
        self.axes_history = deque(maxlen=int(self.smoothing_window))


    def clear(self) -> None:
        self.pose_history.clear()
        self.axes_history.clear()
        self.detected = False
        self.pose = None
        self.raw_cam_pos = None
        self.axes = None
        self.finger_values = None


    def clear_histories(self) -> None:
        self.pose_history.clear()
        self.axes_history.clear()


    def has_start_offset(self, eps: float = 1e-6) -> bool:
        return bool(np.linalg.norm(self.start_offset_cam) > eps)


    @staticmethod
    def average_quaternions_wxyz(quats_wxyz: np.ndarray) -> np.ndarray:
        """Average quaternions with sign alignment. Input shape (N,4) as (qw,qx,qy,qz)."""
        if quats_wxyz.shape[0] == 1:
            q = quats_wxyz[0]
            return q / np.linalg.norm(q)

        q_ref = quats_wxyz[0]
        aligned = quats_wxyz.copy()
        dots = np.sum(aligned * q_ref, axis=1)
        aligned[dots < 0] *= -1.0

        q_mean = np.mean(aligned, axis=0)
        norm = np.linalg.norm(q_mean)
        if norm < 1e-12:
            return q_ref / np.linalg.norm(q_ref)
        return q_mean / norm


    def smooth_pose(self) -> Optional[tuple]:
        if not self.pose_history:
            return None

        arr = np.array(self.pose_history, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 7:
            return tuple(self.pose_history[-1])

        pos_mean = np.mean(arr[:, 0:3], axis=0)
        quat_mean = self.average_quaternions_wxyz(arr[:, 3:7])

        extras = ()
        if arr.shape[1] > 7:
            extras_mean = np.mean(arr[:, 7:], axis=0)
            extras = tuple(float(x) for x in extras_mean.tolist())

        return (
            float(pos_mean[0]), float(pos_mean[1]), float(pos_mean[2]),
            float(quat_mean[0]), float(quat_mean[1]), float(quat_mean[2]), float(quat_mean[3]),
            *extras,
        )


    def smooth_axes(self):
        if not self.axes_history:
            return None

        # Sign-align axes in the window to the most recent sample.
        # This prevents the mean from collapsing when an occasional frame flips an axis direction.
        ref_origin, ref_x, ref_y, _ref_z = self.axes_history[-1]
        aligned_xs = []
        aligned_ys = []

        origins = np.array([a[0] for a in self.axes_history], dtype=np.float64)  # (N,2)
        origin_mean = np.mean(origins, axis=0)
        origin = (int(origin_mean[0]), int(origin_mean[1]))

        for _o, x, y, _z in self.axes_history:
            x2 = x
            y2 = y
            if float(np.dot(x2, ref_x)) < 0.0:
                x2 = -x2
            if float(np.dot(y2, ref_y)) < 0.0:
                y2 = -y2
            aligned_xs.append(x2)
            aligned_ys.append(y2)

        xs = np.mean(np.array(aligned_xs, dtype=np.float64), axis=0)
        ys = np.mean(np.array(aligned_ys, dtype=np.float64), axis=0)

        x_norm = np.linalg.norm(xs)
        y_norm = np.linalg.norm(ys)
        if x_norm < 1e-9 or y_norm < 1e-9:
            return self.axes_history[-1]
        x_axis = xs / x_norm
        y_axis = ys / y_norm

        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-9:
            return self.axes_history[-1]
        z_axis /= z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        return (origin, x_axis, y_axis, z_axis)


    def update_smoothed_outputs(self) -> None:
        if self.detected and self.pose_history:
            self.pose = self.smooth_pose()
            self.axes = self.smooth_axes() if self.axes_history else None
        else:
            self.pose = None
            self.axes = None


    @staticmethod
    def map_finger_value(val, min_val, max_val):
        if val < min_val:
            return 0.0
        if val > max_val:
            return 1.0
        return (val - min_val) / (max_val - min_val)


    @staticmethod
    def finger_angle(im_lm, base_idx):
        base_vec = np.array([im_lm[base_idx + 1].x, im_lm[base_idx + 1].y, im_lm[base_idx + 1].z]) \
                 - np.array([im_lm[base_idx].x, im_lm[base_idx].y, im_lm[base_idx].z])
        tip_vec = np.array([im_lm[base_idx + 3].x, im_lm[base_idx + 3].y, im_lm[base_idx + 3].z]) \
                - np.array([im_lm[base_idx + 2].x, im_lm[base_idx + 2].y, im_lm[base_idx + 2].z])
        angle = np.arccos(np.clip(np.dot(base_vec, tip_vec) /
                         (np.linalg.norm(base_vec) * np.linalg.norm(tip_vec) + 1e-8), -1.0, 1.0))
        return 3.14 - angle


    def calculate_finger_values(self, im_lm) -> Tuple[float, float, float]:
        self.finger_values = self.compute_finger_values(im_lm)
        return self.finger_values


    @classmethod
    def compute_finger_values(cls, im_lm) -> Tuple[float, float, float]:
        thumb_angle = cls.finger_angle(im_lm, 1)
        index_angle = cls.finger_angle(im_lm, 5)
        pinky_angle = cls.finger_angle(im_lm, 17)

        index_v = float(cls.map_finger_value(index_angle, 1.0, 2.0))
        pinky_v = float(cls.map_finger_value(pinky_angle, 1.0, 2.0))
        thumb_v = float(cls.map_finger_value(thumb_angle, 1.5, 2.0))
        return (index_v, pinky_v, thumb_v)