#! /usr/bin/env python3
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from dataclasses import dataclass
import socket
import yaml
from pathlib import Path
from hand_data import HandData

import threading
import time

class VideoCaptureAsync:
    def __init__(self, video_source):
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)  # RTSP - MediaMTX

        self.cap = cap
        print('\n')

        time.sleep(0.5)
        # Try to open stream quickly
        if not self.cap.isOpened():
            raise RuntimeError(
                "❌ Unable to open video source.\n"
                "Make sure webcam/MediaMTX is streaming before running this script.\n"
                f"Source: {video_source}"
            )

        # Try grabbing a first frame to ensure it's really alive
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError(
                "❌ Stream opened but no frames received.\n"
                "Start publishing the stream first (webcam/MediaMTX)."
            )

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Reduce internal frame buffering to 1 frame.
        # This minimizes latency (important for real-time tracking),
        # but may increase the chance of dropped frames if processing is slow.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Request ~30 FPS input rate.
        # # Acts as a hint; real FPS is determined by the camera/stream and may differ.
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.succeeded = False
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            self.succeeded, frame = self.cap.read()
            if self.succeeded:
                self.frame = frame

    def read(self):
        return self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        return self.cap.release()


@dataclass
class FrameDetection:
    cam_pos_n: np.ndarray
    raw_cam_pos: np.ndarray
    finger_values: Tuple[float, float, float]
    rel_rot_mat_a: Optional[np.ndarray]
    rel_rot_mat_b: Optional[np.ndarray]
    axes_a: Optional[Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray]]
    axes_b: Optional[Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray]]
    order_x: int
    handedness: Optional[str]
    handedness_score: float


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


    def __init__(self, udp_ip='127.0.0.1', udp_port=5005, video_source='0'):
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
        
        # Pitch angles (adjustable via UI sliders)
        self.camera_pitch_deg = 15.0  # Camera tilt upward
        self.robot_frame_pitch_deg = -60.0  # Robot frame pitch
        self.camera_pitch_rotation = R.from_euler('x', self.camera_pitch_deg, degrees=True).as_matrix()
        self.robot_frame_pitch_rotation = R.from_euler('x', self.robot_frame_pitch_deg, degrees=True).as_matrix()

        # Persistent per-hand containers (updated every frame)
        self.smoothing_window = 10
        self.left_hand = HandData('left', smoothing_window=self.smoothing_window)
        self.right_hand = HandData('right', smoothing_window=self.smoothing_window)

        # Set Isaac -> Mujoco rotation matrices
        self.left_hand.isaac_to_mujoco_rot = np.array(
            [[0, 0, 1],
             [-1, 0, 0],
             [0, -1, 0]]
        )
        self.right_hand.isaac_to_mujoco_rot = np.array(
            [[0, 0, -1],
             [1, 0, 0],
             [0, -1, 0]]
        )

        # UDP setup
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Video input setup
        self.video_source = video_source

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

        self.left_hand.robot_offset = np.array([-0.15, 0.2, 0.85], dtype=np.float64)
        self.right_hand.robot_offset = np.array([0.15, 0.2, 0.85], dtype=np.float64)

        # Trackbar visibility state
        self.trackbars_visible = False
        self.saved_trackbar_values = {}
        
        # Overlay visibility state
        self.overlay_visible = False
        
        # Parameter file path
        self.params_file = Path('handtracking_params.yaml')
        
        # Initialize calibration state before loading (may be set to True by load_parameters)
        self.is_calibrated = False
        
        # Load parameters from file if it exists
        self.load_parameters()

        # Gesture states
        self.is_activated = False
        self.is_recording = False
        self.is_reset = True

        self.callback_number = 0
        self.prev_callback_number = 0


    @staticmethod
    def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < eps:
            return None
        return v / n


    @staticmethod
    def estimate_plane_normal(points: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        """Estimate a stable normal of the best-fit plane through 3D points.

        Returns a unit vector (sign-ambiguous). Uses SVD on the centered point cloud.
        """
        if points is None:
            return None
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
            return None
        if not np.all(np.isfinite(pts)):
            return None

        c = np.mean(pts, axis=0)
        a = pts - c
        # Degenerate if all points are nearly identical.
        if float(np.linalg.norm(a)) < eps:
            return None

        # Right singular vector with smallest singular value is plane normal.
        try:
            _u, _s, vh = np.linalg.svd(a, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        n = vh[-1]
        return HandTracking.safe_normalize(n, eps=eps)


    @staticmethod
    def rotation_angle_rad(rot_mat: np.ndarray) -> float:
        """Angle of a rotation matrix (0..pi)."""
        tr = float(np.trace(rot_mat))
        c = (tr - 1.0) * 0.5
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.arccos(c))


    def choose_rel_rot_mat(
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
        da = self.rotation_angle_rad(prev_rel.T @ cand_a)
        db = self.rotation_angle_rad(prev_rel.T @ cand_b)
        return (cand_b, True) if db < da else (cand_a, False)


    @staticmethod
    def extract_handedness_and_score(landmark_data, idx: int) -> Tuple[Optional[str], float]:
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
    def with_xyz(pose: tuple, xyz: np.ndarray) -> tuple:
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]), *pose[3:])


    def cam_norm_to_robot_m(
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

        if self.cm_per_px_1 is None or self.cm_per_px_2 is None:
            return np.zeros(3, dtype=np.float64)

        z1, z2 = self.ref_dist_1, self.ref_dist_2
        c1, c2 = self.cm_per_px_1, self.cm_per_px_2
        if z2 == z1:
            cm_per_px = c1
        else:
            cm_per_px = c1 + (c2 - c1) * (z_cm - z1) / (z2 - z1)
        if cm_per_px is None:
            return np.zeros(3, dtype=np.float64)

        wrist_px_x = x_n * float(self.image_w)
        wrist_px_y = y_n * float(self.image_h)

        # Preserve original left/right anchoring behavior
        anchor = 0.35 if side == 'left' else 0.65
        x_cm = -cm_per_px * (wrist_px_x - anchor * float(self.image_w))
        y_cm = -cm_per_px * (wrist_px_y - float(self.image_h) / 2.0)

        # Camera-frame position in cm (X: lateral, Y: vertical, Z: depth)
        cam_pos_cm = np.array([-x_cm, self.ref_dist_2 - z_cm, y_cm], dtype=np.float64)
        
        # Apply camera pitch correction
        cam_pos_cm = self.camera_pitch_rotation @ cam_pos_cm
        
        # Convert to meters and apply scaling
        isaac_robot = np.array(
            [
                x_scaling * (cam_pos_cm[0] / 100.0),
                y_scaling * (cam_pos_cm[1] / 100.0),
                z_scaling * (cam_pos_cm[2] / 100.0),
            ],
            dtype=np.float64,
        )

        # Apply Isaac -> Mujoco rotations
        if side == 'left':
            return self.left_hand.isaac_to_mujoco_rot @ isaac_robot
        else:
            return self.right_hand.isaac_to_mujoco_rot @ isaac_robot


    def calibrate_step_1(self, im_lm, hand_size_px, rot_mat):
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


    def calibrate_step_2(self, im_lm, hand_size_px):
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
        self.is_calibrated = True
        print("Calibration Step 2 completed")
        self.save_parameters()


    def quaternion_distance(self, q1, q2):
        """Compute a distance metric between two quaternions (in wxyz format)."""
        q1 = np.array(q1)
        q2 = np.array(q2)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        angle = 2 * np.arccos(dot)
        return angle


    def check_start_position(
            self,
            target_x: float = 0.5,
            target_x_dist: float = 0.3,
            min_y: float = 0.4) -> bool:
        if self.left_hand.pose is None or self.right_hand.pose is None:
            return False

        left_hand = self.left_hand.pose
        right_hand = self.right_hand.pose

        avg_x = (left_hand[0] + right_hand[0]) / 2
        avg_y = (left_hand[1] + right_hand[1]) / 2
        dist_x = abs(left_hand[0] - right_hand[0])

        # print("Start Gesture Check")
        # print(f"\tX error: {abs(avg_x - target_x):.2f}, Y: {avg_y:.2f} > {min_y}")

        return (abs(avg_x - target_x) < 0.2 and
                abs(dist_x - target_x_dist) < 0.2 and
                avg_y > min_y)


    def check_activate_teleop_gesture(self) -> bool:
        # Open hand in start position
        if not (self.check_start_position() and 
                self.left_hand.finger_values is not None and 
                self.right_hand.finger_values is not None):
            return False
        all_fingers = [*self.left_hand.finger_values[:-1], *self.right_hand.finger_values[:-1]]
        return np.mean(all_fingers) < 0.2


    def check_record_gesture(self) -> bool:
        # Closed hand in start position
        if not (self.check_start_position() and 
                self.left_hand.finger_values is not None and 
                self.right_hand.finger_values is not None):
            return False
        all_fingers = [*self.left_hand.finger_values[:-1], *self.right_hand.finger_values[:-1]]
        return np.mean(all_fingers) > 0.8


    def check_save_gesture(self) -> bool:
        if not (self.left_hand.finger_values is not None and 
                self.right_hand.finger_values is not None):
            return False
        all_fingers = [*self.left_hand.finger_values[:-1], *self.right_hand.finger_values[:-1]]
        return np.mean(all_fingers) < 0.2


    def check_reset_gesture(self) -> bool:
        if not (self.left_hand.finger_values is not None and 
                self.right_hand.finger_values is not None):
            return False
        all_fingers = [*self.left_hand.finger_values[:-1], *self.right_hand.finger_values[:-1]]
        return np.mean(all_fingers) > 0.8


    def check_discard_gesture(
            self,
            target_x: float = 0.5,
            min_x_dist: float = 0.5,
            max_y: float = 0.4,
            target_quat_left: tuple = (0.51, 0.86, 0.03, -0.06),
            target_quat_right: tuple = (-0.04, -0.31, 0.84, -0.44)) -> bool:
        '''
        Left Hand:
                x=0.14 y=0.45 z=0.78
                Qw=-0.00 Qx=1.00 Qy=0.01 Qz=-0.02
        Right Hand:
                x=0.89 y=0.45 z=0.81
                Qw=0.15 Qx=-0.23 Qy=0.96 Qz=0.03

        Left Hand:
                x=0.20 y=0.32 z=2.15
                Qw=0.51 Qx=0.86 Qy=0.03 Qz=-0.06
                Index=1.000 Pinky=1.000 Thumb=1.000
        Right Hand:
                x=0.87 y=0.31 z=2.11
                Qw=-0.04 Qx=-0.31 Qy=0.84 Qz=-0.44
                Index=1.000 Pinky=1.000 Thumb=1.000
        '''
        if self.left_hand.pose is None or self.right_hand.pose is None:
            return False

        left_hand = self.left_hand.pose
        right_hand = self.right_hand.pose

        avg_x = (left_hand[0] + right_hand[0]) / 2
        avg_y = (left_hand[1] + right_hand[1]) / 2
        dist_x = abs(left_hand[0] - right_hand[0])
        # quat_left = [left_hand[3], left_hand[4], left_hand[5], left_hand[6]]
        # quat_right = [right_hand[3], right_hand[4], right_hand[5], right_hand[6]]

        # print("Discard Gesture Check")
        # print(f"\tX error: {abs(avg_x - target_x):.2f}, Y: {avg_y:.2f}")
        # print(f"\tLeft quat dist: {self.quaternion_distance(quat_left, target_quat_left):.2f}, Right quat dist: {self.quaternion_distance(quat_right, target_quat_right):.2f}")

        left_fv = self.left_hand.finger_values
        right_fv = self.right_hand.finger_values
        if left_fv is None or right_fv is None:
            return False

        # self.quaternion_distance(quat_left, target_quat_left) < 0.7 and
        # self.quaternion_distance(quat_right, target_quat_right) < 0.7 and

        # Raise hand up with open palm to discard
        return avg_y < max_y and \
               all(fv > 0.5 for fv in (*left_fv, *right_fv))


    @staticmethod
    def pos_cost(a: np.ndarray, b: np.ndarray, z_weight: float = 0.5) -> float:
        """Distance cost for association in camera-normalized space."""
        da = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        if da.shape[0] >= 3:
            da = np.array([da[0], da[1], da[2] * z_weight], dtype=np.float64)
        return float(np.linalg.norm(da))


    def assign_detections_temporal(
        self,
        detections: list[FrameDetection],
        prev_left: Optional[np.ndarray],
        prev_right: Optional[np.ndarray],
    ) -> list[Optional[FrameDetection]]:
        """Assign detections to (left,right) using handedness + temporal position continuity.

        Goal: avoid random left/right switching by preferring the assignment that minimizes
        movement relative to the previous frame's positions.
        """
        assigned: list[Optional[FrameDetection]] = [None, None]
        if not detections:
            return assigned

        # If we only have one detection, bind it to the closest previously-tracked hand.
        if len(detections) == 1:
            det = detections[0]
            if prev_left is not None and prev_right is not None:
                dl = self.pos_cost(det.cam_pos_n, prev_left)
                dr = self.pos_cost(det.cam_pos_n, prev_right)
                assigned[0 if dl <= dr else 1] = det
                return assigned
            # Fall back to handedness when available.
            if det.handedness in ('left', 'right'):
                assigned[0 if det.handedness == 'left' else 1] = det
            else:
                # Final fallback: x-order relative to image center.
                assigned[0 if det.order_x < int((self.image_w or 0) * 0.5) else 1] = det
            return assigned

        # Keep at most two detections (num_hands=2). Prefer higher handedness confidence when available.
        dets = detections
        if len(dets) > 2:
            dets = sorted(dets, key=lambda d: float(d.handedness_score), reverse=True)[:2]

        d0, d1 = dets[0], dets[1]

        # If we have both previous positions, choose the assignment with smaller total motion.
        if prev_left is not None and prev_right is not None:
            # Position-only costs.
            cost_keep = self.pos_cost(d0.cam_pos_n, prev_left) + self.pos_cost(d1.cam_pos_n, prev_right)
            cost_swap = self.pos_cost(d0.cam_pos_n, prev_right) + self.pos_cost(d1.cam_pos_n, prev_left)

            # Add a small penalty when explicit handedness disagrees with the slot.
            def handedness_penalty(det: FrameDetection, slot: int) -> float:
                if det.handedness not in ('left', 'right'):
                    return 0.0
                want = 'left' if slot == 0 else 'right'
                if det.handedness == want:
                    return 0.0
                # Higher score => stronger disagreement => larger penalty.
                return 0.15 + 0.25 * float(det.handedness_score)

            cost_keep += handedness_penalty(d0, 0) + handedness_penalty(d1, 1)
            cost_swap += handedness_penalty(d0, 1) + handedness_penalty(d1, 0)

            # Hysteresis: only swap if it is meaningfully better.
            swap_margin = 0.08
            if cost_swap + swap_margin < cost_keep:
                assigned[0], assigned[1] = d1, d0
            else:
                assigned[0], assigned[1] = d0, d1
            return assigned

        # If we only have one previous position, bind the closest detection to that side.
        if prev_left is not None and prev_right is None:
            dl0 = self.pos_cost(d0.cam_pos_n, prev_left)
            dl1 = self.pos_cost(d1.cam_pos_n, prev_left)
            if dl0 <= dl1:
                assigned[0], assigned[1] = d0, d1
            else:
                assigned[0], assigned[1] = d1, d0
            return assigned
        if prev_right is not None and prev_left is None:
            dr0 = self.pos_cost(d0.cam_pos_n, prev_right)
            dr1 = self.pos_cost(d1.cam_pos_n, prev_right)
            if dr0 <= dr1:
                assigned[1], assigned[0] = d0, d1
            else:
                assigned[1], assigned[0] = d1, d0
            return assigned

        # No temporal info: fall back to stable ordering by handedness then x.
        # Place by handedness if possible.
        for det in dets:
            if det.handedness not in ('left', 'right'):
                continue
            slot = 0 if det.handedness == 'left' else 1
            if assigned[slot] is None:
                assigned[slot] = det
            else:
                if float(det.handedness_score) > float(assigned[slot].handedness_score):
                    assigned[slot] = det

        # Fill remaining slots by x-order.
        used_ids = set(id(d) for d in assigned if d is not None)
        remaining = [d for d in dets if id(d) not in used_ids]
        remaining_sorted = sorted(remaining, key=lambda d: d.order_x)
        for det in remaining_sorted:
            if assigned[0] is None:
                assigned[0] = det
            elif assigned[1] is None:
                assigned[1] = det
        return assigned


    def create_trackbars(self, window_name):
        """Create all trackbars in the window."""
        def _noop(_val):
            pass

        scale_mult = 100
        scale_max = 300
        
        # Get saved values or use defaults
        horiz_val = self.saved_trackbar_values.get('Horiz', int(1.0 * scale_mult))
        vert_val = self.saved_trackbar_values.get('Vert', int(1.0 * scale_mult))
        depth_val = self.saved_trackbar_values.get('Depth', int(1.0 * scale_mult))
        cam_pitch_val = self.saved_trackbar_values.get('CamPitch', int(self.camera_pitch_deg + 90))
        robot_pitch_val = self.saved_trackbar_values.get('RobotPitch', int(self.robot_frame_pitch_deg + 90))
        
        cv2.createTrackbar('Horiz', window_name, horiz_val, scale_max, _noop)
        cv2.createTrackbar('Vert', window_name, vert_val, scale_max, _noop)
        cv2.createTrackbar('Depth', window_name, depth_val, scale_max, _noop)
        cv2.createTrackbar('CamPitch', window_name, cam_pitch_val, 180, _noop)
        cv2.createTrackbar('RobotPitch', window_name, robot_pitch_val, 180, _noop)


    def destroy_trackbars(self, window_name):
        """Destroy all trackbars and save their current values."""
        trackbar_names = ['Horiz', 'Vert', 'Depth', 'CamPitch', 'RobotPitch']
        for name in trackbar_names:
            try:
                self.saved_trackbar_values[name] = cv2.getTrackbarPos(name, window_name)
            except:
                pass
        
        # OpenCV doesn't have a direct way to remove trackbars, so we recreate the window
        cv2.destroyWindow(window_name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)


    def toggle_trackbars(self, window_name):
        """Toggle trackbar visibility."""
        self.trackbars_visible = not self.trackbars_visible
        if self.trackbars_visible:
            self.create_trackbars(window_name)
            print("Trackbars: ON")
        else:
            self.destroy_trackbars(window_name)
            print("Trackbars: OFF")


    def toggle_overlay(self):
        """Toggle text overlay visibility."""
        self.overlay_visible = not self.overlay_visible
        print(f"Overlay: {'ON' if self.overlay_visible else 'OFF'}")


    def save_parameters(self):
        """Save current parameters to YAML file."""
        params = {
            'horiz': self.saved_trackbar_values.get('Horiz', 100),
            'vert': self.saved_trackbar_values.get('Vert', 100),
            'depth': self.saved_trackbar_values.get('Depth', 100),
            'cam_pitch': self.saved_trackbar_values.get('CamPitch', int(self.camera_pitch_deg + 90)),
            'robot_pitch': self.saved_trackbar_values.get('RobotPitch', int(self.robot_frame_pitch_deg + 90)),
        }
        
        # Save calibration data if calibrated
        if self.is_calibrated:
            params['calibration'] = {
                'palm_sizes_1': self.palm_sizes_1,
                'cm_per_px_1': float(self.cm_per_px_1) if self.cm_per_px_1 is not None else None,
                'ref_rot_mat': self.ref_rot_mat.tolist() if self.ref_rot_mat is not None else None,
                'palm_sizes_2': self.palm_sizes_2,
                'f_times_H_edges': self.f_times_H_edges,
                'cm_per_px_2': float(self.cm_per_px_2) if self.cm_per_px_2 is not None else None,
            }
        
        try:
            with open(self.params_file, 'w') as f:
                yaml.dump(params, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save parameters: {e}")


    def load_parameters(self):
        """Load parameters from YAML file if it exists."""
        if not self.params_file.exists():
            return
        
        try:
            with open(self.params_file, 'r') as f:
                params = yaml.safe_load(f)
            
            if params:
                self.saved_trackbar_values['Horiz'] = params.get('horiz', 100)
                self.saved_trackbar_values['Vert'] = params.get('vert', 100)
                self.saved_trackbar_values['Depth'] = params.get('depth', 100)
                self.saved_trackbar_values['CamPitch'] = params.get('cam_pitch', int(self.camera_pitch_deg + 90))
                self.saved_trackbar_values['RobotPitch'] = params.get('robot_pitch', int(self.robot_frame_pitch_deg + 90))
                print(f"Loaded parameters from {self.params_file}")
                
                # Load calibration data if available
                if 'calibration' in params:
                    cal = params['calibration']
                    self.palm_sizes_1 = cal.get('palm_sizes_1')
                    self.cm_per_px_1 = cal.get('cm_per_px_1')
                    ref_rot_mat_list = cal.get('ref_rot_mat')
                    if ref_rot_mat_list is not None:
                        self.ref_rot_mat = np.array(ref_rot_mat_list, dtype=np.float64)
                    self.palm_sizes_2 = cal.get('palm_sizes_2')
                    self.f_times_H_edges = cal.get('f_times_H_edges')
                    self.cm_per_px_2 = cal.get('cm_per_px_2')
                    
                    # Check if all calibration parameters are present
                    if all([
                        self.palm_sizes_1 is not None,
                        self.cm_per_px_1 is not None,
                        self.ref_rot_mat is not None,
                        self.palm_sizes_2 is not None,
                        self.f_times_H_edges is not None,
                        self.cm_per_px_2 is not None
                    ]):
                        self.is_calibrated = True
                        print("Auto-calibrated from saved parameters")
        except Exception as e:
            print(f"Warning: Could not load parameters: {e}")


    def tracking_loop(self):
        cap = None
        local_camera = False
        if args.video_source.isdigit():
            print(f"Opening local camera at index {args.video_source}...")
            cap = cv2.VideoCapture(int(args.video_source))  # Local camera
            local_camera = True
        else:
            cap = VideoCaptureAsync(args.video_source)

        frame_timestamp_ms = 0
        window_name = 'Hand Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # Create initial trackbars
        scale_mult = 100
        # self.create_trackbars(window_name)

        while cap.isOpened():
            if local_camera:
                success, image = cap.read()
                if not success:
                    print("Failed to read from camera. Exiting.")
                    break
            else:
                image = cap.read()
                if not cap.succeeded:
                    break

            # Per-frame detections; assign them to (left, right)
            frame_detections: list[FrameDetection] = []

            if self.mirror:
                image = cv2.flip(image, 1)

            frame_timestamp_ms += 33  # ~30 FPS

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            landmark_data = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            annotated = image.copy()

            # Read scaling factors from UI (use saved values if trackbars hidden)
            params_changed = False
            if self.trackbars_visible:
                x_scaling = cv2.getTrackbarPos('Horiz', window_name) / scale_mult
                y_scaling = cv2.getTrackbarPos('Vert', window_name) / scale_mult
                z_scaling = cv2.getTrackbarPos('Depth', window_name) / scale_mult
                
                # Read and update pitch angles from UI
                camera_pitch_deg = cv2.getTrackbarPos('CamPitch', window_name) - 90
                robot_pitch_deg = cv2.getTrackbarPos('RobotPitch', window_name) - 90
                
                # Check if values changed and update saved values
                new_horiz = int(x_scaling * scale_mult)
                new_vert = int(y_scaling * scale_mult)
                new_depth = int(z_scaling * scale_mult)
                new_cam_pitch = camera_pitch_deg + 90
                new_robot_pitch = robot_pitch_deg + 90
                
                if (self.saved_trackbar_values.get('Horiz') != new_horiz or
                    self.saved_trackbar_values.get('Vert') != new_vert or
                    self.saved_trackbar_values.get('Depth') != new_depth or
                    self.saved_trackbar_values.get('CamPitch') != new_cam_pitch or
                    self.saved_trackbar_values.get('RobotPitch') != new_robot_pitch):
                    params_changed = True
                
                self.saved_trackbar_values['Horiz'] = new_horiz
                self.saved_trackbar_values['Vert'] = new_vert
                self.saved_trackbar_values['Depth'] = new_depth
                self.saved_trackbar_values['CamPitch'] = new_cam_pitch
                self.saved_trackbar_values['RobotPitch'] = new_robot_pitch
                
                # Save to file if parameters changed
                if params_changed:
                    self.save_parameters()

                # Display current scaling values and pitch angles (if overlay visible)
                if self.overlay_visible:
                    overlay_lines = [
                        f"Horiz: {x_scaling:.2f}    Vert: {y_scaling:.2f}    Depth: {z_scaling:.2f}",
                        f"Camera Pitch: {self.camera_pitch_deg:.1f}deg    Wrist Pitch: {self.robot_frame_pitch_deg:.1f}deg",
                    ]
                    y0 = 30
                    for i, text in enumerate(overlay_lines):
                        org = (10, y0 + i * 25)
                        cv2.putText(annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Use saved values when trackbars are hidden
                x_scaling = self.saved_trackbar_values.get('Horiz', 100) / scale_mult
                y_scaling = self.saved_trackbar_values.get('Vert', 100) / scale_mult
                z_scaling = self.saved_trackbar_values.get('Depth', 100) / scale_mult
                camera_pitch_deg = self.saved_trackbar_values.get('CamPitch', int(self.camera_pitch_deg + 90)) - 90
                robot_pitch_deg = self.saved_trackbar_values.get('RobotPitch', int(self.robot_frame_pitch_deg + 90)) - 90
            
            if abs(camera_pitch_deg - self.camera_pitch_deg) > 0.1:
                self.camera_pitch_deg = camera_pitch_deg
                self.camera_pitch_rotation = R.from_euler('x', self.camera_pitch_deg, degrees=True).as_matrix()
            if abs(robot_pitch_deg - self.robot_frame_pitch_deg) > 0.1:
                self.robot_frame_pitch_deg = robot_pitch_deg
                self.robot_frame_pitch_rotation = R.from_euler('x', self.robot_frame_pitch_deg, degrees=True).as_matrix()

            primary_im_lm = None
            primary_hand_size_px = None
            primary_rot_mat = None

            if landmark_data.hand_world_landmarks:
                for idx, w_lm in enumerate(landmark_data.hand_world_landmarks):
                    self.image_h, self.image_w, _ = image.shape

                    det_side, det_side_score = self.extract_handedness_and_score(landmark_data, idx)
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
                    ring_mcp   = np.array([w_lm[13].x, w_lm[13].y, w_lm[13].z])
                    pinky_mcp  = np.array([w_lm[17].x, w_lm[17].y, w_lm[17].z])

                    vec_middle = middle_mcp - wrist
                    vec_pinky_index = index_mcp - pinky_mcp

                    # Deterministic basis construction (enforced directions):
                    #   Y axis: wrist -> middle finger (towards middle_mcp)
                    #   X axis: pinky -> index (towards index_mcp)
                    #   Z axis: palm normal from right-hand rule
                    # This removes the x/z flip ambiguity, which could otherwise cause visible z flips.
                    y_axis = self.safe_normalize(vec_middle)
                    rot_cam_a = None
                    rot_cam_b = None
                    rot_mat_a = None
                    rot_mat_b = None
                    axes_a_world = None
                    axes_b_world = None

                    if y_axis is not None:
                        # Single robust rule:
                        # - Build a deterministic right-handed basis when geometry is well-conditioned.
                        # - If it becomes degenerate (e.g., fingertips toward camera), use a plane-fit normal.
                        # - If plane-fit is used, consider both normal signs and let temporal continuity choose.
                        basis_eps = 1e-6

                        # Try to get X from (pinky->index) projected onto the plane orthogonal to Y.
                        x_proj = vec_pinky_index - y_axis * float(np.dot(vec_pinky_index, y_axis))
                        x_axis = self.safe_normalize(x_proj, eps=basis_eps)

                        if x_axis is not None:
                            # Deterministic Z from right-hand rule.
                            z_axis = self.safe_normalize(np.cross(x_axis, y_axis), eps=basis_eps)
                            if z_axis is not None:
                                # Re-orthogonalize X to reduce drift.
                                x_axis = self.safe_normalize(np.cross(y_axis, z_axis), eps=basis_eps)
                                if x_axis is not None:
                                    rot_cam_a = np.column_stack((x_axis, y_axis, z_axis))
                                    axes_a_world = (x_axis, y_axis, z_axis)
                        else:
                            # Fallback: plane-fit normal from palm points.
                            palm_pts = np.stack([wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0)
                            n = self.estimate_plane_normal(palm_pts)
                            if n is not None:
                                # Remove any component along Y to ensure orthogonality.
                                n = n - y_axis * float(np.dot(n, y_axis))
                                n = self.safe_normalize(n, eps=basis_eps)

                            if n is not None:
                                # Candidate A: use n
                                x_a = self.safe_normalize(np.cross(y_axis, n), eps=basis_eps)
                                if x_a is not None:
                                    z_a = self.safe_normalize(np.cross(x_a, y_axis), eps=basis_eps)
                                    if z_a is not None:
                                        rot_cam_a = np.column_stack((x_a, y_axis, z_a))
                                        axes_a_world = (x_a, y_axis, z_a)

                                # Candidate B: use -n
                                x_b = self.safe_normalize(np.cross(y_axis, -n), eps=basis_eps)
                                if x_b is not None:
                                    z_b = self.safe_normalize(np.cross(x_b, y_axis), eps=basis_eps)
                                    if z_b is not None:
                                        rot_cam_b = np.column_stack((x_b, y_axis, z_b))
                                        axes_b_world = (x_b, y_axis, z_b)

                    if rot_cam_a is not None:
                        rot_mat_a = self.reorient_mat @ rot_cam_a
                    if rot_cam_b is not None:
                        rot_mat_b = self.reorient_mat @ rot_cam_b

                    wrist_px   = np.array([im_lm[0].x * self.image_w,   im_lm[0].y * self.image_h])
                    middle_px  = np.array([im_lm[9].x * self.image_w,  im_lm[9].y * self.image_h])
                    hand_size_px = np.linalg.norm(wrist_px - middle_px)

                    if idx == 0:
                        primary_im_lm = im_lm
                        primary_hand_size_px = hand_size_px
                        primary_rot_mat = rot_mat_a

                    if self.is_calibrated:
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

                        # Apply rotations to transform from camera to robot frame
                        rel_rot_mat_a = None
                        rel_rot_mat_b = None
                        if rot_mat_a is not None and self.ref_rot_mat is not None:
                            rel_rot_mat_a = rot_mat_a @ self.ref_rot_mat.T
                            rel_rot_mat_a = self.robot_frame_change_basis @ rel_rot_mat_a @ self.robot_frame_change_basis.T
                            rel_rot_mat_a = self.robot_frame_rotation @ rel_rot_mat_a
                            rel_rot_mat_a = self.robot_frame_pitch_rotation @ rel_rot_mat_a

                        if rot_mat_b is not None and self.ref_rot_mat is not None:
                            rel_rot_mat_b = rot_mat_b @ self.ref_rot_mat.T
                            rel_rot_mat_b = self.robot_frame_change_basis @ rel_rot_mat_b @ self.robot_frame_change_basis.T
                            rel_rot_mat_b = self.robot_frame_rotation @ rel_rot_mat_b
                            rel_rot_mat_b = self.robot_frame_pitch_rotation @ rel_rot_mat_b

                        # Visualize selected palm edge
                        conn = self.PALM_CONNECTIONS[best_idx]
                        start = im_lm[conn[0]]
                        end   = im_lm[conn[1]]
                        cv2.line(annotated,
                                    (int(start.x * self.image_w), int(start.y * self.image_h)),
                                    (int(end.x * self.image_w),   int(end.y * self.image_h)),
                                    (0, 1, 0), 2 * self.THICKNESS)

                        # Cache axes for drawing after temporal disambiguation
                        origin = (int(im_lm[0].x * self.image_w), int(im_lm[0].y * self.image_h))

                        # Axes for visualization; match chosen temporal candidate.
                        axes_a = None
                        axes_b = None
                        if axes_a_world is not None:
                            xa, ya, za = axes_a_world
                            axes_a = (origin, xa, ya, za)
                        if axes_b_world is not None:
                            xb, yb, zb = axes_b_world
                            axes_b = (origin, xb, yb, zb)

                        # Compute finger values per-detection (assigned to HandData after left/right association)
                        fv_tuple = HandData.compute_finger_values(im_lm)

                        frame_detections.append(
                            FrameDetection(
                                cam_pos_n=cam_pos_n,
                                raw_cam_pos=cam_pos_n,
                                finger_values=tuple(float(v) for v in fv_tuple),
                                rel_rot_mat_a=rel_rot_mat_a,
                                rel_rot_mat_b=rel_rot_mat_b,
                                axes_a=axes_a,
                                axes_b=axes_b,
                                order_x=int(origin[0]),
                                handedness=det_side,
                                handedness_score=float(det_side_score),
                            )
                        )

            # Temporal association to prevent random left/right switching.
            prev_left = self.left_hand.raw_cam_pos if self.left_hand.detected else None
            prev_right = self.right_hand.raw_cam_pos if self.right_hand.detected else None
            assigned = self.assign_detections_temporal(frame_detections, prev_left, prev_right)

            # Moving-average smoothing over last N frames (per hand index)
            hands = [self.left_hand, self.right_hand]
            for i in range(2):
                det = assigned[i]
                if det is not None:
                    raw_pos = det.raw_cam_pos
                    chosen_rel, chose_flip = self.choose_rel_rot_mat(
                        hands[i].prev_rel_rot_mat,
                        det.rel_rot_mat_a,
                        det.rel_rot_mat_b,
                    )

                    if chosen_rel is None:
                        # No stable rotation available yet.
                        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                    else:
                        # Apply hand-specific axis corrections
                        if i == 1:  # Right hand only
                            # Pitch: +x → -x, Roll: +y unchanged, Yaw: +z → -z
                            right_correction = np.array([
                                [-1,  0,  0],
                                [ 0,  1,  0],
                                [ 0,  0, -1]
                            ], dtype=np.float64)
                            chosen_rel = chosen_rel @ right_correction
                        
                        quat = R.from_matrix(chosen_rel).as_quat()  # [x, y, z, w]
                        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

                    cam_pos_n = det.cam_pos_n
                    fv = det.finger_values
                    axes = det.axes_b if chose_flip else det.axes_a

                    hands[i].ingest_frame(
                        cam_pos_n=cam_pos_n,
                        quat_wxyz=(qw, qx, qy, qz),
                        finger_values=fv,
                        raw_cam_pos=raw_pos,
                        axes=axes,
                        chosen_rel_rot_mat=chosen_rel,
                    )
                else:
                    # Detection lost - preserve last position instead of clearing
                    hands[i].mark_not_detected()

            # Produce smoothed outputs (only if we have enough info)
            for i in range(2):
                hands[i].update_smoothed_outputs()

            activate_gesture = self.check_activate_teleop_gesture()
            record_gesture = self.check_record_gesture()
            save_gesture = self.check_save_gesture()
            discard_gesture = self.check_discard_gesture()
            reset_gesture = self.check_reset_gesture()

            if self.left_hand.pose is not None or self.right_hand.pose is not None:
                # Check for start, save, and discard gestures
                if not self.is_activated and self.is_reset and activate_gesture:
                    # ACTIVATE
                    self.callback_number = 1
                    self.is_activated = True
                elif self.is_activated and not self.is_recording and self.is_reset and record_gesture:
                    # START RECORDING
                    self.callback_number = 2
                    self.is_recording = True
                elif self.is_recording and save_gesture:
                    # SAVE RECORDING
                    self.callback_number = 3
                    self.is_recording = False
                    self.is_reset = False
                elif self.is_recording and discard_gesture:
                    # DISCARD RECORDING
                    self.callback_number = 4
                    self.is_recording = False
                    self.is_reset = False
                elif self.is_activated and not self.is_recording and reset_gesture and not discard_gesture:
                    # RESET
                    self.callback_number = 5
                    self.is_reset = True
                    self.is_activated = False

                # Draw smoothed axes
                scale = 80
                for axes in [self.left_hand.axes, self.right_hand.axes]:
                    if axes is None:
                        continue
                    origin, x_axis, y_axis, z_axis = axes
                    for axis, color in zip([x_axis, y_axis, z_axis], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        end_pt = (int(origin[0] + axis[0] * scale), int(origin[1] + axis[1] * scale))
                        cv2.arrowedLine(annotated, origin, end_pt, color, 3, tipLength=0.2)

                # Display handedness label near each wrist (if overlay visible)
                if self.overlay_visible:
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
                    # output += f"\n\tx={hand[0]:.2f} y={hand[1]:.2f} z={hand[2]:.2f}"
                    # output += f"\n\tQw={hand[3]:.2f} Qx={hand[4]:.2f} Qy={hand[5]:.2f} Qz={hand[6]:.2f}"
                    # output += f"\n\tR={hand_rpy[0]:.1f} P={hand_rpy[1]:.1f} Y={hand_rpy[2]:.1f}\n"
                    output += f"\n\tIndex={hand[7]:.3f} Pinky={hand[8]:.3f} Thumb={hand[9]:.3f}"
                    # print(output)

                if self.callback_number > 0:
                    self.prev_callback_number = self.callback_number
                    # print(f"Sending callback number: {self.callback_number}")

                # Send UDP message with hand pose data + callback number
                default_pose_cam = (0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
                left_cam = self.left_hand.pose if self.left_hand.pose is not None else default_pose_cam
                right_cam = self.right_hand.pose if self.right_hand.pose is not None else default_pose_cam

                # Convert CAMERA -> ROBOT at the very end (right before UDP send)
                left_cam_xyz = np.array(left_cam[0:3], dtype=np.float64)
                right_cam_xyz = np.array(right_cam[0:3], dtype=np.float64)
                left_robot_raw = self.cam_norm_to_robot_m(left_cam_xyz, 'left', x_scaling, y_scaling, z_scaling)
                right_robot_raw = self.cam_norm_to_robot_m(right_cam_xyz, 'right', x_scaling, y_scaling, z_scaling)

                if self.left_hand.has_start_offset():
                    start_left_robot = self.cam_norm_to_robot_m(self.left_hand.start_offset_cam, 'left', x_scaling, y_scaling, z_scaling)
                    left_robot = left_robot_raw - start_left_robot + self.left_hand.robot_offset
                else:
                    left_robot = left_robot_raw + self.left_hand.robot_offset

                if self.right_hand.has_start_offset():
                    start_right_robot = self.cam_norm_to_robot_m(self.right_hand.start_offset_cam, 'right', x_scaling, y_scaling, z_scaling)
                    right_robot = right_robot_raw - start_right_robot + self.right_hand.robot_offset
                else:
                    right_robot = right_robot_raw + self.right_hand.robot_offset

                # On the START frame, force the reported xyz to equal the IsaacSim offsets exactly.
                if self.active_gesture == 'start' and self.callback_number == 1 and self.left_hand.pose is not None and self.right_hand.pose is not None:
                    left_robot = self.left_hand.robot_offset.astype(np.float64)
                    right_robot = self.right_hand.robot_offset.astype(np.float64)

                # Reorder quaternion from (w,x,y,z) to (x,y,z,w) format
                # left_cam/right_cam format: (x, y, z, qw, qx, qy, qz, finger1, finger2, finger3)
                left_udp = (float(left_robot[0]), float(left_robot[1]), float(left_robot[2]), 
                           left_cam[4], left_cam[5], left_cam[6], left_cam[3], *left_cam[7:])
                right_udp = (float(right_robot[0]), float(right_robot[1]), float(right_robot[2]), 
                            right_cam[4], right_cam[5], right_cam[6], right_cam[3], *right_cam[7:])
                pose_data = left_udp + right_udp + (float(self.callback_number),)

                msg = np.array(pose_data, dtype=np.float32).tobytes()
                self.sock.sendto(msg, (self.udp_ip, self.udp_port))

                # self.callback_number = 0

                # Display gesture text
                if self.prev_callback_number == 1:
                    cv2.putText(annotated, 'ACTIVATED', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
                elif self.prev_callback_number == 2:
                    cv2.putText(annotated, 'RECORDING', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
                elif self.prev_callback_number == 3:
                    cv2.putText(annotated, 'SAVED', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                elif self.prev_callback_number == 4:
                    cv2.putText(annotated, 'DISCARDED', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('0'):
                self.mirror = not self.mirror
                print(f"Mirror: {'ON' if self.mirror else 'OFF'}")
            elif key == ord('1') and primary_im_lm is not None:
                self.calibrate_step_1(primary_im_lm, primary_hand_size_px, primary_rot_mat)
            elif key == ord('2') and primary_im_lm is not None and primary_hand_size_px is not None:
                self.calibrate_step_2(primary_im_lm, primary_hand_size_px)
            elif key == ord('t') or key == ord('T'):
                self.toggle_trackbars(window_name)
            elif key == ord('o') or key == ord('O'):
                self.toggle_overlay()

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--udp_ip', type=str, default='127.0.0.1', help='UDP target IP')
    parser.add_argument('--udp_port', type=int, default=5005, help='UDP target port')
    # parser.add_argument('--video_source', type=str, default='0', help='Camera index or RTSP URL (default: local webcam)')
    parser.add_argument('--video_source', type=str, default='rtsp://127.0.0.1:8554/webcam?rtsp_transport=udp', help='Camera index or RTSP URL (default: local MediaMTX server)')
    args = parser.parse_args()
    ht = HandTracking(udp_ip=args.udp_ip, udp_port=args.udp_port)
    ht.tracking_loop()
