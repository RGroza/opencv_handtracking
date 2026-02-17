import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import socket


@dataclass
class HandData:
    """Per-hand state container updated every frame."""

    side: str  # 'left' | 'right'
    detected: bool = False

    # Smoothed pose tuple in CAMERA-NORMALIZED coordinates:
    # (x_n, y_n, z_n, qw, qx, qy, qz, index, pinky, thumb)
    # where x_n,y_n are the MediaPipe image landmark coords in [0,1]
    # and z_n is normalized depth (not clipped) with:
    #   z_n=0  -> distance at _calibrate_step_1 (ref_dist_1)
    #   z_n=1  -> distance at _calibrate_step_2 (ref_dist_2)
    # Values <0 or >1 mean extrapolation outside the calibration range.
    pose: Optional[tuple] = None

    # Raw (pre-smoothing) camera-normalized position (x_n, y_n, z_n) for this frame.
    raw_cam_pos: Optional[np.ndarray] = None

    # Smoothed axes in image space: (origin_px, x_axis, y_axis, z_axis)
    axes: Optional[Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray]] = None

    # Previous (last selected) relative rotation matrix used for temporal disambiguation.
    prev_rel_rot_mat: Optional[np.ndarray] = None


class HandTracking:
    MODEL_PATH = "hand_landmarker.task"
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    PALM_CONNECTIONS = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 17)]
    LANDMARK_COLOR = (255, 255, 255)
    CONNECTION_COLOR = (0, 0, 0)
    PALM_CONNECTION_COLOR = (0, 1, 0)
    THICKNESS = 2
    RADIUS = 4


    def __init__(self, udp_ip='127.0.0.1', udp_port=5005):
        self.mirror = True
        self.image_w = None
        self.image_h = None

        self.hand_height_cm = 17.0  # measured real hand size cm (wrist to middle tip)
        self.ref_dist_1 = 20.0      # cm
        self.ref_dist_2 = 50.0      # cm
        self.cm_per_px_1 = None
        self.cm_per_px_2 = None
        self.palm_sizes_1 = None
        self.palm_sizes_2 = None
        self.f_times_H_edges = None
        self.ref_rot_mat = None

        self.calibrated = False
        self.callback_number = 0

        # Fixed rotation from camera frame (C) to robot base frame (B)
        # Camera is in facing the user while the robot's perspective is from it's head
        # Axes mapping: x_B = -x_C, y_B = -z_C, z_B = +y_C
        self.reorient_mat = np.array(
            [[-1, 0, 0],
             [0, 0, -1],
             [0, 1, 0]]
        )
        self.robot_frame_rotation = np.diag([1, -1, -1])
        self.robot_frame_change_basis = np.diag([1, 1, -1])

        # Persistent per-hand containers (updated every frame)
        self.left_hand = HandData('left')
        self.right_hand = HandData('right')
        self.hand_poses = []
        self.wrist_axes = []
        self.finger_values = []

        # Smoothing (moving average)
        self.smoothing_window = 10
        # Histories are indexed after left/right ordering (0=left, 1=right)
        self.pose_histories = [deque(maxlen=self.smoothing_window) for _ in range(2)]
        self.axis_histories = [deque(maxlen=self.smoothing_window) for _ in range(2)]

        # UDP setup
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        BaseOptions = python.BaseOptions
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        HandLandmarker = vision.HandLandmarker
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.active_gesture = None  # None | 'start' | 'end'
        self.gesture_step = 0
        self.override_gesture = False
        self.episode_started = False

        self.robot_left_offset = np.array([-0.15, 0.2, 0.85])
        self.robot_right_offset = np.array([0.15, 0.2, 0.85])
        self.start_left_offset = np.zeros(3)
        self.start_right_offset = np.zeros(3)


    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < eps:
            return None
        return v / n


    @staticmethod
    def _rotation_angle_rad(rot_mat: np.ndarray) -> float:
        """Angle of a rotation matrix (0..pi)."""
        tr = float(np.trace(rot_mat))
        c = (tr - 1.0) * 0.5
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.arccos(c))


    def _choose_rel_rot_mat(
        self,
        prev_rel: Optional[np.ndarray],
        cand_a: Optional[np.ndarray],
        cand_b: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Pick the candidate closest to prev_rel. Returns (chosen, chose_b)."""
        if cand_a is None and cand_b is None:
            return (prev_rel, False)
        if prev_rel is None:
            return (cand_a if cand_a is not None else cand_b, cand_a is None)
        if cand_a is None:
            return (cand_b, True)
        if cand_b is None:
            return (cand_a, False)

        # Choose by smallest relative rotation angle.
        da = self._rotation_angle_rad(prev_rel.T @ cand_a)
        db = self._rotation_angle_rad(prev_rel.T @ cand_b)
        return (cand_b, True) if db < da else (cand_a, False)


    @staticmethod
    def _extract_handedness_and_score(landmark_data, idx: int) -> Tuple[Optional[str], float]:
        """Return ('left'|'right'|None, score)."""
        handedness = getattr(landmark_data, 'handedness', None)
        if not handedness or idx >= len(handedness):
            return (None, 0.0)

        cls_list = handedness[idx]
        categories = getattr(cls_list, 'categories', None)
        if categories is None:
            # Some versions expose a plain list
            categories = cls_list
        if not categories:
            return (None, 0.0)

        cat0 = categories[0]
        name = getattr(cat0, 'category_name', None) or getattr(cat0, 'display_name', None)
        score = float(getattr(cat0, 'score', 0.0) or 0.0)
        if not name:
            return (None, score)

        name_l = str(name).strip().lower()
        if 'left' in name_l:
            return ('left', score)
        if 'right' in name_l:
            return ('right', score)
        return (None, score)


    @staticmethod
    def _with_xyz(pose: tuple, xyz: np.ndarray) -> tuple:
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]), *pose[3:])


    def _cam_norm_to_robot_m(
        self,
        cam_xyz_n: np.ndarray,
        side: str,
        x_scaling: float,
        y_scaling: float,
        z_scaling: float,
    ) -> np.ndarray:
        """Convert (x_n,y_n,z_n) to robot base frame meters using calibration at the end."""
        if self.image_w is None or self.image_h is None:
            return np.zeros(3, dtype=np.float64)

        x_n, y_n, z_n = float(cam_xyz_n[0]), float(cam_xyz_n[1]), float(cam_xyz_n[2])
        z_cm = float(self.ref_dist_1 + z_n * (self.ref_dist_2 - self.ref_dist_1))
        cm_per_px = self._cm_per_px(z_cm)
        if cm_per_px is None:
            return np.zeros(3, dtype=np.float64)

        wrist_px_x = x_n * float(self.image_w)
        wrist_px_y = y_n * float(self.image_h)

        # Preserve original left/right anchoring behavior
        anchor = 0.35 if side == 'left' else 0.65
        x_cm = -cm_per_px * (wrist_px_x - anchor * float(self.image_w))
        y_cm = -cm_per_px * (wrist_px_y - float(self.image_h) / 2.0)

        # Convert position from camera frame to robot base frame in meters
        # (x_R = -z_C, y_R = -x_C, z_R = +y_C)
        return np.array(
            [
                x_scaling * (-x_cm / 100.0),
                y_scaling * ((self.ref_dist_2 - z_cm) / 100.0),
                z_scaling * (y_cm / 100.0),
            ],
            dtype=np.float64,
        )


    @staticmethod
    def _average_quaternions_wxyz(quats_wxyz: np.ndarray) -> np.ndarray:
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


    def _smooth_pose(self, pose_history: deque) -> tuple:
        arr = np.array(pose_history, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 7:
            return tuple(pose_history[-1])

        pos_mean = np.mean(arr[:, 0:3], axis=0)
        quat_mean = self._average_quaternions_wxyz(arr[:, 3:7])

        # Preserve and smooth finger values
        extras = ()
        if arr.shape[1] > 7:
            extras_mean = np.mean(arr[:, 7:], axis=0)
            extras = tuple(float(x) for x in extras_mean.tolist())

        return (
            float(pos_mean[0]), float(pos_mean[1]), float(pos_mean[2]),
            float(quat_mean[0]), float(quat_mean[1]), float(quat_mean[2]), float(quat_mean[3]),
            *extras,
        )


    def _smooth_axes(self, axes_history: deque):
        if not axes_history:
            return None

        origins = np.array([a[0] for a in axes_history], dtype=np.float64)  # (N,2)
        origin_mean = np.mean(origins, axis=0)
        origin = (int(origin_mean[0]), int(origin_mean[1]))

        xs = np.mean(np.array([a[1] for a in axes_history], dtype=np.float64), axis=0)
        ys = np.mean(np.array([a[2] for a in axes_history], dtype=np.float64), axis=0)

        x_norm = np.linalg.norm(xs)
        y_norm = np.linalg.norm(ys)
        if x_norm < 1e-9 or y_norm < 1e-9:
            return axes_history[-1]
        x_axis = xs / x_norm
        y_axis = ys / y_norm

        # Re-orthonormalize
        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-9:
            return axes_history[-1]
        z_axis /= z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        return (origin, x_axis, y_axis, z_axis)


    def _cm_per_px(self, z_cm):
        if self.cm_per_px_1 is None or self.cm_per_px_2 is None:
            return None
        z1, z2 = self.ref_dist_1, self.ref_dist_2
        c1, c2 = self.cm_per_px_1, self.cm_per_px_2
        if z2 == z1:
            return c1
        return c1 + (c2 - c1) * (z_cm - z1) / (z2 - z1)


    def _calibrate_step_1(self, im_lm, hand_size_px, rot_mat):
        if self.image_w is None or self.image_h is None:
            print("Image size not set yet; wait for a frame before calibrating.")
            return

        self.palm_sizes_1 = []
        for connection in self.PALM_CONNECTIONS:
            start = im_lm[connection[0]]
            end = im_lm[connection[1]]
            dx = (start.x - end.x) * self.image_w
            dy = (start.y - end.y) * self.image_h
            self.palm_sizes_1.append(float(np.hypot(dx, dy)))

        self.cm_per_px_1 = self.hand_height_cm / hand_size_px
        self.ref_rot_mat = rot_mat
        print("Calibration Step 1 completed")


    def _calibrate_step_2(self, im_lm, hand_size_px):
        if self.image_w is None or self.image_h is None:
            print("Image size not set yet; wait for a frame before calibrating.")
            return
        if not self.palm_sizes_1:
            print("Run calibration step 1 first")
            return

        self.palm_sizes_2 = []
        for connection in self.PALM_CONNECTIONS:
            start = im_lm[connection[0]]
            end = im_lm[connection[1]]
            dx = (start.x - end.x) * self.image_w
            dy = (start.y - end.y) * self.image_h
            self.palm_sizes_2.append(float(np.hypot(dx, dy)))

        z1 = float(self.ref_dist_1)
        z2 = float(self.ref_dist_2)
        self.f_times_H_edges = [
            (s1 * z1 + s2 * z2) / 2.0 for s1, s2 in zip(self.palm_sizes_1, self.palm_sizes_2)
        ]

        self.cm_per_px_2 = self.hand_height_cm / hand_size_px
        self.calibrated = True
        print("Calibration Step 2 completed")


    def _finger_value(self, val, min_val, max_val):
        if val < min_val:
            return 0.0
        elif val > max_val:
            return 1.0
        else:
            return (val - min_val) / (max_val - min_val)


    def _quaternion_distance(self, q1, q2):
        """Compute a distance metric between two quaternions (in wxyz format)."""
        q1 = np.array(q1)
        q2 = np.array(q2)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        angle = 2 * np.arccos(dot)
        return angle


    def check_start_gesture(
            self,
            target_x: float = 0.5,
            target_x_dist: float = 0.3,
            min_y: float = 0.4,
            target_quat_left: tuple = (0.47, 0.73, -0.27, 0.41),
            target_quat_right: tuple = (0.51, 0.56, -0.55, 0.35)) -> bool:
        '''
        Left Hand:
                x=0.34 y=0.88 z=0.28
                Qw=0.47 Qx=0.73 Qy=-0.27 Qz=0.41
        Right Hand:
                x=0.65 y=0.85 z=0.34
                Qw=0.51 Qx=0.56 Qy=-0.55 Qz=0.35
        '''
        if self.left_hand.pose is None or self.right_hand.pose is None:
            return False
        
        left_hand = self.left_hand.pose
        right_hand = self.right_hand.pose

        avg_x = (left_hand[0] + right_hand[0]) / 2
        avg_y = (left_hand[1] + right_hand[1]) / 2
        dist_x = abs(left_hand[0] - right_hand[0])
        quat_left = [left_hand[3], left_hand[4], left_hand[5], left_hand[6]]
        quat_right = [right_hand[3], right_hand[4], right_hand[5], right_hand[6]]
    
        # print("Start Gesture Check")
        # print(f"\tX error: {abs(avg_x - target_x):.2f}, Y: {avg_y:.2f} > {min_y}")
        # print(f"\tLeft quat dist: {self._quaternion_distance(quat_left, target_quat_left):.2f}, Right quat dist: {self._quaternion_distance(quat_right, target_quat_right):.2f}")

        return (abs(avg_x - target_x) < 0.2 and
                abs(dist_x - target_x_dist) < 0.2 and
                avg_y > min_y and
                self._quaternion_distance(quat_left, target_quat_left) < 1.5 and
                self._quaternion_distance(quat_right, target_quat_right) < 1.5 and
                all(fv > 0.5 for fv in self.finger_values))


    def check_stop_gesture(
            self,
            target_x: float = 0.5,
            min_x_dist: float = 0.6,
            max_y: float = 0.4,
            target_quat_left: tuple = (0.0, 1.0, 0.0, 0.0),
            target_quat_right: tuple = (0.0, 0.0, 1.0, 0.0)) -> bool:
        '''
        Left Hand:
                x=0.14 y=0.45 z=0.78
                Qw=-0.00 Qx=1.00 Qy=0.01 Qz=-0.02
        Right Hand:
                x=0.89 y=0.45 z=0.81
                Qw=0.15 Qx=-0.23 Qy=0.96 Qz=0.03
        '''
        if self.left_hand.pose is None or self.right_hand.pose is None:
            return False
        
        left_hand = self.left_hand.pose
        right_hand = self.right_hand.pose

        avg_x = (left_hand[0] + right_hand[0]) / 2
        avg_y = (left_hand[1] + right_hand[1]) / 2
        dist_x = abs(left_hand[0] - right_hand[0])
        quat_left = [left_hand[3], left_hand[4], left_hand[5], left_hand[6]]
        quat_right = [right_hand[3], right_hand[4], right_hand[5], right_hand[6]]

        # print("Stop Gesture Check")
        # print(f"\tX error: {abs(avg_x - target_x):.2f}, Y: {avg_y:.2f}")
        # print(f"\tLeft quat dist: {self._quaternion_distance(quat_left, target_quat_left):.2f}, Right quat dist: {self._quaternion_distance(quat_right, target_quat_right):.2f}")

        return (abs(avg_x - target_x) < 0.3 and
                dist_x > min_x_dist and
                avg_y < max_y and
                self._quaternion_distance(quat_left, target_quat_left) < 0.7 and
                self._quaternion_distance(quat_right, target_quat_right) < 0.7 and
                all(fv > 0.5 for fv in self.finger_values))


    def tracking_loop(self):
        cap = cv2.VideoCapture(0)
        frame_timestamp_ms = 0
        window_name = 'Hand Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # UI controls: scaling sliders (trackbars)
        # Store scaling as integer hundredths to support float-like control.
        def _noop(_val):
            pass

        scale_mult = 100
        scale_max = 500  # 0.00 .. 5.00
        cv2.createTrackbar('Horiz', window_name, int(2.0 * scale_mult), scale_max, _noop)
        cv2.createTrackbar('Vert', window_name, int(2.0 * scale_mult), scale_max, _noop)
        cv2.createTrackbar('Depth', window_name, int(2.0 * scale_mult), scale_max, _noop)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Per-frame detection list; we will assign them to (left, right)
            detections = []

            if self.mirror:
                image = cv2.flip(image, 1)

            # Read scaling factors from UI (available regardless of calibration state)
            x_scaling = cv2.getTrackbarPos('Horiz', window_name) / scale_mult
            y_scaling = cv2.getTrackbarPos('Vert', window_name) / scale_mult
            z_scaling = cv2.getTrackbarPos('Depth', window_name) / scale_mult

            frame_timestamp_ms += 33  # ~30 FPS

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            landmark_data = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            annotated = image.copy()

            # Display current scaling values
            overlay_lines = [
                f"Horiz: {x_scaling:.2f}",
                f"Vert: {y_scaling:.2f}",
                f"Depth: {z_scaling:.2f}",
            ]
            y0 = 30
            for i, text in enumerate(overlay_lines):
                org = (10, y0 + i * 25)
                cv2.putText(annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            primary_im_lm = None
            primary_hand_size_px = None
            primary_rot_mat = None
            
            if landmark_data.hand_world_landmarks:
                for idx, w_lm in enumerate(landmark_data.hand_world_landmarks):
                    self.image_h, self.image_w, _ = image.shape

                    det_side, det_side_score = self._extract_handedness_and_score(landmark_data, idx)
                    # If we mirror the image, handedness labels swap relative to the real hand.
                    if det_side is not None and self.mirror:
                        det_side = 'right' if det_side == 'left' else 'left'

                    # Draw connections (still using image landmarks for visualization)
                    im_lm = landmark_data.hand_landmarks[idx]
                    for start_idx, end_idx in self.HAND_CONNECTIONS:
                        start = im_lm[start_idx]
                        end = im_lm[end_idx]
                        start_px = (int(start.x * self.image_w), int(start.y * self.image_h))
                        end_px = (int(end.x * self.image_w), int(end.y * self.image_h))
                        cv2.line(annotated, start_px, end_px, self.CONNECTION_COLOR, self.THICKNESS)

                    # Draw landmarks (image space for visualization)
                    for lm in im_lm:
                        px = (int(lm.x * self.image_w), int(lm.y * self.image_h))
                        cv2.circle(annotated, px, self.RADIUS, self.LANDMARK_COLOR, -1)

                    # ────────────────────────────────────────────────
                    # Orientation from WORLD landmarks (metric 3D)
                    # ────────────────────────────────────────────────
                    wrist      = np.array([w_lm[0].x,  w_lm[0].y,  w_lm[0].z])
                    index_mcp  = np.array([w_lm[5].x,  w_lm[5].y,  w_lm[5].z])
                    middle_mcp = np.array([w_lm[9].x,  w_lm[9].y,  w_lm[9].z])
                    pinky_mcp  = np.array([w_lm[17].x, w_lm[17].y, w_lm[17].z])

                    vec_middle = middle_mcp - wrist
                    vec_pinky_index = index_mcp - pinky_mcp

                    # Robust basis construction: skip/hold when the geometry is ill-conditioned.
                    y_axis = self._safe_normalize(vec_middle)
                    z_axis = None
                    x_axis = None
                    rot_cam = None
                    rot_mat = None
                    rot_cam_flip = None
                    rot_mat_flip = None

                    if y_axis is not None:
                        z_axis = self._safe_normalize(np.cross(vec_pinky_index, y_axis))
                        if z_axis is not None:
                            x_axis = self._safe_normalize(np.cross(y_axis, z_axis))
                    if x_axis is not None and y_axis is not None and z_axis is not None:
                        rot_cam = np.column_stack((x_axis, y_axis, z_axis))
                        rot_mat = self.reorient_mat @ rot_cam

                        # The palm normal has a 180° sign ambiguity. Flipping x and z keeps det(+1).
                        flip_xz = np.diag([-1.0, 1.0, -1.0])
                        rot_cam_flip = rot_cam @ flip_xz
                        rot_mat_flip = self.reorient_mat @ rot_cam_flip

                    wrist_px   = np.array([im_lm[0].x * self.image_w,   im_lm[0].y * self.image_h])
                    middle_px  = np.array([im_lm[9].x * self.image_w,  im_lm[9].y * self.image_h])
                    hand_size_px = np.linalg.norm(wrist_px - middle_px)

                    if idx == 0:
                        primary_im_lm = im_lm
                        primary_hand_size_px = hand_size_px
                        primary_rot_mat = rot_mat

                    if self.calibrated:
                        # Foreshortening-robust depth:
                        # Compute a depth estimate from each palm edge and take the median
                        if not self.palm_sizes_1 or not self.f_times_H_edges:
                            continue

                        curr_sizes = []
                        est_z_values = []
                        for p_idx, _ref_size in enumerate(self.palm_sizes_1):
                            conn = self.PALM_CONNECTIONS[p_idx]
                            start = im_lm[conn[0]]
                            end = im_lm[conn[1]]
                            dx = (start.x - end.x) * self.image_w
                            dy = (start.y - end.y) * self.image_h
                            curr_size = float(np.hypot(dx, dy))
                            curr_sizes.append(curr_size)

                            if curr_size > 1e-6:
                                est_z_values.append(self.f_times_H_edges[p_idx] / curr_size)

                        if not est_z_values:
                            continue

                        # Get the minimum estimated depth from the palm edges and its index (for visualization)
                        est_z = min(est_z_values)  # pseudo-depth (cm)
                        best_idx = np.argmin(est_z_values)

                        # Camera-normalized pose for smoothing + gesture detection
                        x_n = float(im_lm[0].x)
                        y_n = float(im_lm[0].y)
                        denom = float(self.ref_dist_2 - self.ref_dist_1)
                        z_n = 0.0 if abs(denom) < 1e-9 else float((est_z - self.ref_dist_1) / denom)
                        cam_pos_n = np.array([x_n, y_n, z_n], dtype=np.float64)

                        rel_rot_mat = None
                        rel_rot_mat_flip = None
                        if rot_mat is not None and self.ref_rot_mat is not None:
                            rel_rot_mat = rot_mat @ self.ref_rot_mat.T
                            rel_rot_mat = self.robot_frame_change_basis @ rel_rot_mat @ self.robot_frame_change_basis.T
                            rel_rot_mat = self.robot_frame_rotation @ rel_rot_mat
                        if rot_mat_flip is not None and self.ref_rot_mat is not None:
                            rel_rot_mat_flip = rot_mat_flip @ self.ref_rot_mat.T
                            rel_rot_mat_flip = self.robot_frame_change_basis @ rel_rot_mat_flip @ self.robot_frame_change_basis.T
                            rel_rot_mat_flip = self.robot_frame_rotation @ rel_rot_mat_flip

                        # Calculate midpoint of wrist, middle_mcp, and pinky_mcp
                        palm_center = (wrist + middle_mcp + pinky_mcp) / 3.0
                        thumb_tip = np.array([w_lm[4].x, w_lm[4].y, w_lm[4].z])
                        index_tip = np.array([w_lm[8].x, w_lm[8].y, w_lm[8].z])
                        pinky_tip = np.array([w_lm[20].x, w_lm[20].y, w_lm[20].z])
                        # Calculate distance of each fingertip to the palm center
                        wrist_palm_dist = np.linalg.norm(palm_center - wrist)
                        thumb_dist = np.linalg.norm(thumb_tip - palm_center) / wrist_palm_dist
                        index_dist = np.linalg.norm(index_tip - palm_center) / wrist_palm_dist
                        pinky_dist = np.linalg.norm(pinky_tip - palm_center) / wrist_palm_dist

                        # print(f"Hand {idx}: Wrist-Palm Dist={wrist_palm_dist:.3f}, Thumb={thumb_dist:.3f}, Index={index_dist:.3f}, Pinky={pinky_dist:.3f}")

                        self.finger_values = [self._finger_value(index_dist, 1.4, 1.9),
                                              self._finger_value(pinky_dist, 1.2, 1.5),
                                              self._finger_value(thumb_dist, 0.8, 1.6)]

                        # Visualize selected palm edge
                        conn = self.PALM_CONNECTIONS[best_idx]
                        start = im_lm[conn[0]]
                        end   = im_lm[conn[1]]
                        cv2.line(annotated,
                                    (int(start.x * self.image_w), int(start.y * self.image_h)),
                                    (int(end.x * self.image_w),   int(end.y * self.image_h)),
                                    (0, 1, 0), 2 * self.THICKNESS)

                        # Cache axes for drawing after stability/jump check
                        origin = (int(im_lm[0].x * self.image_w), int(im_lm[0].y * self.image_h))

                        # Axes for visualization; also keep a flipped variant consistent with rot_mat_flip.
                        axes = None
                        axes_flip = None
                        if x_axis is not None and y_axis is not None and z_axis is not None:
                            axes = (origin, x_axis, y_axis, z_axis)
                            axes_flip = (origin, -x_axis, y_axis, -z_axis)

                        detections.append({
                            'cam_pos_n': cam_pos_n,
                            'finger_values': tuple(float(v) for v in self.finger_values),
                            'rel_rot_mat': rel_rot_mat,
                            'rel_rot_mat_flip': rel_rot_mat_flip,
                            'raw_cam_pos': cam_pos_n,
                            'axes': axes,
                            'axes_flip': axes_flip,
                            'order_x': origin[0],
                            'handedness': det_side,
                            'handedness_score': float(det_side_score),
                            'origin': origin,
                        })

            # Assign detections to (left, right) ordering using handedness when available.
            assigned = [None, None]  # 0=left, 1=right

            # First pass: place by handedness
            for det in detections:
                side = det.get('handedness')
                if side not in ('left', 'right'):
                    continue
                slot = 0 if side == 'left' else 1
                if assigned[slot] is None:
                    assigned[slot] = det
                else:
                    # Tie-break by higher handedness confidence
                    if float(det.get('handedness_score', 0.0)) > float(assigned[slot].get('handedness_score', 0.0)):
                        assigned[slot] = det

            # Second pass: fill remaining slots by x-order with unassigned detections
            used_ids = set(id(d) for d in assigned if d is not None)
            remaining = [d for d in detections if id(d) not in used_ids]
            remaining_sorted = sorted(remaining, key=lambda d: d['order_x'])
            for det in remaining_sorted:
                if assigned[0] is None:
                    assigned[0] = det
                elif assigned[1] is None:
                    assigned[1] = det
                else:
                    break

            # Moving-average smoothing over last N frames (per hand index)
            hands = [self.left_hand, self.right_hand]
            for i in range(2):
                det = assigned[i]
                if det is not None:
                    raw_pos = det['raw_cam_pos']

                    chosen_rel, chose_flip = self._choose_rel_rot_mat(
                        hands[i].prev_rel_rot_mat,
                        det.get('rel_rot_mat'),
                        det.get('rel_rot_mat_flip'),
                    )

                    if chosen_rel is None:
                        # No stable rotation available yet.
                        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                    else:
                        quat = R.from_matrix(chosen_rel).as_quat()  # [x, y, z, w]
                        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                        hands[i].prev_rel_rot_mat = chosen_rel

                    cam_pos_n = det['cam_pos_n']
                    fv = det['finger_values']
                    pose = (
                        float(cam_pos_n[0]), float(cam_pos_n[1]), float(cam_pos_n[2]),
                        float(qw), float(qx), float(qy), float(qz),
                        float(fv[0]), float(fv[1]), float(fv[2]),
                    )

                    axes = det.get('axes_flip') if chose_flip else det.get('axes')

                    self.pose_histories[i].append(pose)
                    if axes is not None:
                        self.axis_histories[i].append(axes)

                    hands[i].detected = True
                    hands[i].raw_cam_pos = raw_pos
                else:
                    self.pose_histories[i].clear()
                    self.axis_histories[i].clear()

                    hands[i].detected = False
                    hands[i].pose = None
                    hands[i].raw_cam_pos = None
                    hands[i].axes = None
                    hands[i].prev_rel_rot_mat = None

            # Produce smoothed outputs (only if we have enough info)
            for i in range(2):
                if hands[i].detected and len(self.pose_histories[i]) > 0:
                    hands[i].pose = self._smooth_pose(self.pose_histories[i])
                    hands[i].axes = self._smooth_axes(self.axis_histories[i]) if len(self.axis_histories[i]) > 0 else None

            # Also keep legacy lists in sync (some downstream code expects these)
            self.hand_poses = [h.pose for h in hands if h.pose is not None]
            self.wrist_axes = [h.axes for h in hands if h.axes is not None]

            if self.left_hand.pose is not None or self.right_hand.pose is not None:
                # Check for start and end gestures
                start_active = self.check_start_gesture()
                end_active = (not start_active) and self.check_stop_gesture()

                if start_active:
                    # Keep START text visible while gesture persists.
                    cv2.putText(annotated, 'START', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    if self.active_gesture != 'start':
                        self.active_gesture = 'start'
                        self.gesture_step = 0

                    # Emit: 1 -> 0 (once per sustained detection)
                    if not self.episode_started:
                        if self.gesture_step == 0:
                            self.callback_number = 1
                            self.gesture_step = 1
                            # Store starting offsets in CAMERA frame (meters). Robot conversion happens at UDP pack time.
                            if self.left_hand.raw_cam_pos is not None and self.right_hand.raw_cam_pos is not None:
                                self.start_left_offset = self.left_hand.raw_cam_pos
                                self.start_right_offset = self.right_hand.raw_cam_pos
                            print(f"Start gesture detected. Left offset: {self.start_left_offset}, Right offset: {self.start_right_offset}")

                            # Reset smoothing so we don't average pre-START poses into the new rebased frame.
                            for h in self.pose_histories:
                                h.clear()
                            for h in self.axis_histories:
                                h.clear()
                            if len(self.hand_poses) >= 2:
                                # Seed smoothing with the start offsets so the first rebased output is stable.
                                self.pose_histories[0].append(self._with_xyz(self.hand_poses[0], self.start_left_offset))
                                self.pose_histories[1].append(self._with_xyz(self.hand_poses[1], self.start_right_offset))
                        else:
                            self.callback_number = 0
                            self.episode_started = True
                elif end_active:
                    # Keep STOP text visible while gesture persists.
                    cv2.putText(annotated, 'STOP', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    if self.active_gesture != 'stop':
                        self.active_gesture = 'stop'
                        self.gesture_step = 0

                    # Emit: 2 -> 3 -> 0 (once per sustained detection)
                    if self.episode_started:
                        if self.gesture_step == 0:
                            self.callback_number = 2
                            self.gesture_step = 1
                            # Reset the starting hand offsets to zero
                            self.start_left_offset = np.zeros(3)
                            self.start_right_offset = np.zeros(3)
                        elif self.gesture_step == 1:
                            self.callback_number = 3
                            self.gesture_step = 2
                        else:
                            self.callback_number = 0
                            self.episode_started = False
                else:
                    # No gesture detected; reset latch so the next detection replays the sequence.
                    if not self.override_gesture:
                        self.callback_number = 0
                    self.override_gesture = False
                    self.active_gesture = None
                    self.gesture_step = 0

                # Draw smoothed axes
                scale = 80
                for axes in [self.left_hand.axes, self.right_hand.axes]:
                    if axes is None:
                        continue
                    origin, x_axis, y_axis, z_axis = axes
                    for axis, color in zip([x_axis, y_axis, z_axis], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        end_pt = (int(origin[0] + axis[0] * scale), int(origin[1] + axis[1] * scale))
                        cv2.arrowedLine(annotated, origin, end_pt, color, 3, tipLength=0.2)

                # Display handedness label near each wrist
                for label, hand in [('L', self.left_hand), ('R', self.right_hand)]:
                    if hand.pose is None:
                        continue
                    if hand.axes is not None:
                        origin = hand.axes[0]
                    else:
                        origin = (int(hand.pose[0] * self.image_w), int(hand.pose[1] * self.image_h))
                    org = (origin[0] + 12, origin[1] - 12)
                    cv2.putText(annotated, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(annotated, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                # Logging print
                for idx, hand in enumerate([self.left_hand.pose, self.right_hand.pose]):
                    if hand is None:
                        continue
                    hand_rpy = R.from_quat([hand[4], hand[5], hand[6], hand[3]]).as_euler('xyz', degrees=True)
                    output = "Left Hand:" if idx == 0 else "Right Hand:"
                    output += f"\n\tx={hand[0]:.2f} y={hand[1]:.2f} z={hand[2]:.2f}"
                    output += f"\n\tQw={hand[3]:.2f} Qx={hand[4]:.2f} Qy={hand[5]:.2f} Qz={hand[6]:.2f}"
                    # output += f"\n\tR={hand_rpy[0]:.1f} P={hand_rpy[1]:.1f} Y={hand_rpy[2]:.1f}\n"
                    output += f"\n\tIndex={hand[7]:.3f} Pinky={hand[8]:.3f} Thumb={hand[9]:.3f}"
                    print(output)

                if self.callback_number > 0:
                    print(f"Sending callback number: {self.callback_number}")

                # Send UDP message with hand pose data + callback number
                default_pose_cam = (0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
                left_cam = self.left_hand.pose if self.left_hand.pose is not None else default_pose_cam
                right_cam = self.right_hand.pose if self.right_hand.pose is not None else default_pose_cam

                # Convert CAMERA -> ROBOT at the very end (right before UDP send)
                left_cam_xyz = np.array(left_cam[0:3], dtype=np.float64)
                right_cam_xyz = np.array(right_cam[0:3], dtype=np.float64)
                left_robot_raw = self._cam_norm_to_robot_m(left_cam_xyz, 'left', x_scaling, y_scaling, z_scaling)
                right_robot_raw = self._cam_norm_to_robot_m(right_cam_xyz, 'right', x_scaling, y_scaling, z_scaling)

                if np.linalg.norm(self.start_left_offset) > 1e-6:
                    start_left_robot = self._cam_norm_to_robot_m(self.start_left_offset, 'left', x_scaling, y_scaling, z_scaling)
                    left_robot = left_robot_raw - start_left_robot + self.robot_left_offset
                else:
                    left_robot = left_robot_raw + self.robot_left_offset

                if np.linalg.norm(self.start_right_offset) > 1e-6:
                    start_right_robot = self._cam_norm_to_robot_m(self.start_right_offset, 'right', x_scaling, y_scaling, z_scaling)
                    right_robot = right_robot_raw - start_right_robot + self.robot_right_offset
                else:
                    right_robot = right_robot_raw + self.robot_right_offset

                # On the START frame, force the reported xyz to equal the IsaacSim offsets exactly.
                if self.active_gesture == 'start' and self.callback_number == 1 and len(self.hand_poses) >= 2:
                    left_robot = self.robot_left_offset.astype(np.float64)
                    right_robot = self.robot_right_offset.astype(np.float64)

                left_udp = (float(left_robot[0]), float(left_robot[1]), float(left_robot[2]), *left_cam[3:])
                right_udp = (float(right_robot[0]), float(right_robot[1]), float(right_robot[2]), *right_cam[3:])
                pose_data = left_udp + right_udp + (float(self.callback_number),)

                msg = np.array(pose_data, dtype=np.float32).tobytes()
                self.sock.sendto(msg, (self.udp_ip, self.udp_port))

                self.callback_number = 0

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('-') or key == ord('_'):
                self.mirror = not self.mirror
                print(f"Mirror: {'ON' if self.mirror else 'OFF'}")
            elif key == ord('4') and primary_im_lm is not None:
                self._calibrate_step_1(primary_im_lm, primary_hand_size_px, primary_rot_mat)
            elif key == ord('5') and primary_im_lm is not None and primary_hand_size_px is not None:
                self._calibrate_step_2(primary_im_lm, primary_hand_size_px)
            elif key == ord('1'):
                self.callback_number = 1
                self.override_gesture = True
            elif key == ord('2'):
                self.callback_number = 2
                self.override_gesture = True
            elif key == ord('3'):
                self.callback_number = 3
                self.override_gesture = True

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--udp_ip', type=str, default='127.0.0.1', help='UDP target IP')
    parser.add_argument('--udp_port', type=int, default=5005, help='UDP target port')
    args = parser.parse_args()
    ht = HandTracking(udp_ip=args.udp_ip, udp_port=args.udp_port)
    ht.tracking_loop()