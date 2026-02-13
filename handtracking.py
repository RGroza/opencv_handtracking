import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R
from collections import deque


class Handtracking:
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
        self.hand_height_cm = 17.0  # measured real hand size cm (wrist to middle tip)
        self.ref_dist_1 = 20.0      # cm
        self.ref_dist_2 = 50.0      # cm
        self.ref_size_1 = None
        self.ref_size_2 = None
        self.ref_rot_mat = None
        self.cm_per_px_1 = None
        self.cm_per_px_2 = None
        self.f_times_H = None
        self.calibrated = False
        self.callback_number = 0
        self.mirror = True
        self.image_w = None
        self.image_h = None

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

        self.hand_data = None
        self.hand_poses = []
        self.prev_hand_poses = []
        self.wrist_axes = []
        self.prev_wrist_axes = []
        self.palm_sizes_1 = None
        self.palm_sizes_2 = None
        self.f_times_H_edges = None

        # Smoothing (moving average)
        self.smoothing_window = 20
        # Histories are indexed after left/right ordering (0=left, 1=right)
        self.pose_histories = [deque(maxlen=self.smoothing_window) for _ in range(2)]
        self.axis_histories = [deque(maxlen=self.smoothing_window) for _ in range(2)]

        # UDP setup
        import socket
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
    def _with_xyz(pose: tuple, xyz: np.ndarray) -> tuple:
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]), *pose[3:])


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


    def get_cm_per_px(self, z_cm):
        if self.cm_per_px_1 is None or self.cm_per_px_2 is None:
            return None
        z1, z2 = self.ref_dist_1, self.ref_dist_2
        c1, c2 = self.cm_per_px_1, self.cm_per_px_2
        if z2 == z1:
            return c1
        return c1 + (c2 - c1) * (z_cm - z1) / (z2 - z1)


    def calibrate_step1(self, im_lm, hand_size_px, rot_mat):
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


    def calibrate_step2(self, im_lm, hand_size_px):
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


    def map_finger_value(self, val, min_val, max_val):
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
            target_x: float = 0.0,
            target_x_dist: float = 0.3,
            max_z: float = 0.8,
            target_quat_left: tuple = (0.36, 0.76, -0.29, 0.46),
            target_quat_right: tuple = (-0.55, -0.49, 0.56, -0.37)) -> bool:
        '''
        Left Hand:
                x=-0.16 y=0.39 z=1.11
                Qw=0.44 Qx=0.75 Qy=-0.21 Qz=0.44        Thumb=1.000 Index=0.971 Pinky=1.000
        Right Hand:
                x=0.00 y=0.33 z=0.99
                Qw=-0.46 Qx=-0.50 Qy=0.61 Qz=-0.41      Thumb=1.000 Index=0.840 Pinky=1.000
        '''
        if not self.hand_poses or len(self.hand_poses) < 2:
            return False
        
        left_hand = self.hand_poses[0]
        right_hand = self.hand_poses[1]

        avg_x = (left_hand[0] + right_hand[0]) / 2
        avg_z = (left_hand[2] + right_hand[2]) / 2
        dist_x = abs(left_hand[0] - right_hand[0])
        quat_left = [left_hand[3], left_hand[4], left_hand[5], left_hand[6]]
        quat_right = [right_hand[3], right_hand[4], right_hand[5], right_hand[6]]
    
        # print("Start Gesture Check")
        # print(f"\tX error: {abs(avg_x - target_x):.2f}, Z: {avg_z:.2f} < {max_z}")
        # print(f"\tLeft quat dist: {self._quaternion_distance(quat_left, target_quat_left):.2f}, Right quat dist: {self._quaternion_distance(quat_right, target_quat_right):.2f}")

        return (abs(avg_x - target_x) < 0.2 and
                abs(dist_x - target_x_dist) < 0.2 and
                self._quaternion_distance(quat_left, target_quat_left) < 1.5 and
                self._quaternion_distance(quat_right, target_quat_right) < 1.5)


    def check_end_gesture(
            self,
            target_x: float = 0.0,
            min_x_dist: float = 0.7,
            min_z: float = 1.2,
            target_quat_left: tuple = (-0.12, 0.99, -0.10, 0.00),
            target_quat_right: tuple = (0.06, -0.26, 0.96, 0.01)) -> bool:
        '''
        Left Hand:
                x=-0.18 y=0.26 z=0.78
                Qw=-0.12 Qx=0.99 Qy=-0.10 Qz=0.00
                R=-166.7 P=0.8 Y=-11.1
                Thumb=0.956 Index=1.000 Pinky=0.633
        Right Hand:
                x=0.22 y=0.26 z=0.81
                Qw=0.06 Qx=-0.26 Qy=0.96 Qz=0.01
                R=-178.8 P=6.8 Y=-150.1
                Thumb=0.887 Index=1.000 Pinky=0.612
        '''
        if not self.hand_poses or len(self.hand_poses) < 2:
            return False
        
        left_hand = self.hand_poses[0]
        right_hand = self.hand_poses[1]

        avg_x = (left_hand[0] + right_hand[0]) / 2
        avg_z = (left_hand[2] + right_hand[2]) / 2
        dist_x = abs(left_hand[0] - right_hand[0])
        quat_left = [left_hand[3], left_hand[4], left_hand[5], left_hand[6]]
        quat_right = [right_hand[3], right_hand[4], right_hand[5], right_hand[6]]

        # print("End Gesture Check")
        # print(f"\tX error: {abs(avg_x - target_x):.2f}, Z: {avg_z:.2f}")
        # print(f"\tLeft quat dist: {self._quaternion_distance(quat_left, target_quat_left):.2f}, Right quat dist: {self._quaternion_distance(quat_right, target_quat_right):.2f}")

        return (abs(avg_x - target_x) < 0.3 and
                dist_x > min_x_dist and
                avg_z > min_z and
                self._quaternion_distance(quat_left, target_quat_left) < 0.7 and
                self._quaternion_distance(quat_right, target_quat_right) < 0.7)


    def tracking_loop(self):
        cap = cv2.VideoCapture(0)
        frame_timestamp_ms = 0
        cv2.namedWindow('Hand Landmarker (World + Manual Draw)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Landmarker (World + Manual Draw)', 1280, 720)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            self.hand_poses = []
            self.wrist_axes = []
            raw_robot_positions = []  # per detected hand, before applying any offsets

            if self.mirror:
                image = cv2.flip(image, 1)

            frame_timestamp_ms += 33  # ~30 FPS

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            self.hand_data = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            annotated = image.copy()

            primary_im_lm = None
            primary_hand_size_px = None
            primary_rot_mat = None
            
            if self.hand_data.hand_world_landmarks:
                for idx, w_lm in enumerate(self.hand_data.hand_world_landmarks):
                    self.image_h, self.image_w, _ = image.shape

                    # Draw connections (still using image landmarks for visualization)
                    im_lm = self.hand_data.hand_landmarks[idx]
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

                    # Y axis: pointing towards middle finger
                    y_axis = vec_middle / np.linalg.norm(vec_middle)

                    # Z axis: palm normal
                    z_axis = np.cross(vec_pinky_index, y_axis)
                    z_axis /= np.linalg.norm(z_axis)

                    # X axis: pointing towards index finger from pinky
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis /= np.linalg.norm(x_axis)

                    rot_cam = np.column_stack((x_axis, y_axis, z_axis))
                    rot_mat = self.reorient_mat @ rot_cam

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

                        cm_per_px = self.get_cm_per_px(est_z)

                        if idx == 0:
                            x_cm = -cm_per_px * (wrist_px[0] - 0.35 * self.image_w)
                        else:
                            x_cm = -cm_per_px * (wrist_px[0] - 0.65 * self.image_w)
                        y_cm = -cm_per_px * (wrist_px[1] - self.image_h / 2)
                        z_cm = est_z

                        # Convert position from camera frame to robot base frame in meters
                        # (x_R = -z_C, y_R = -x_C, z_R = +y_C)
                        x_scaling = 2.0
                        y_scaling = 2.0
                        z_scaling = 2.0
                        robot_pos = np.array([x_scaling * (-x_cm / 100), y_scaling * (self.ref_dist_2 - z_cm) / 100, z_scaling * (y_cm / 100)])

                        # Keep a copy in a consistent "raw" frame (no artificial offsets)
                        robot_pos_raw = robot_pos.copy()

                        if idx == 0:
                            if np.linalg.norm(self.start_left_offset) > 1e-6:
                                applied = self.robot_left_offset - self.start_left_offset
                                print(f"Applying left hand offset: {applied}")
                                robot_pos = robot_pos_raw - self.start_left_offset + self.robot_left_offset
                            else:
                                robot_pos = robot_pos_raw + self.robot_left_offset
                        else:
                            if np.linalg.norm(self.start_right_offset) > 1e-6:
                                applied = self.robot_right_offset - self.start_right_offset
                                print(f"Applying right hand offset: {applied}")
                                robot_pos = robot_pos_raw - self.start_right_offset + self.robot_right_offset
                            else:
                                robot_pos = robot_pos_raw + self.robot_right_offset

                        rel_rot_mat = rot_mat @ self.ref_rot_mat.T
                        rel_rot_mat = self.robot_frame_change_basis @ rel_rot_mat @ self.robot_frame_change_basis.T
                        rel_rot_mat = self.robot_frame_rotation @ rel_rot_mat
                        quat = R.from_matrix(rel_rot_mat).as_quat()  # [x, y, z, w]
                        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

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

                        finger_values = [self.map_finger_value(index_dist, 0.9, 1.9),
                                         self.map_finger_value(pinky_dist, 0.9, 1.5),
                                         self.map_finger_value(thumb_dist, 0.8, 1.6)]

                        self.hand_poses.append((robot_pos[0], robot_pos[1], robot_pos[2], qw, qx, qy, qz, finger_values[0], finger_values[1], finger_values[2]))
                        raw_robot_positions.append(robot_pos_raw)

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
                        self.wrist_axes.append((origin, x_axis, y_axis, z_axis))

            if self.hand_poses:
                # Left/right hand separation logic (based on x in image space)
                if len(self.hand_poses) > 1:
                    if self.hand_poses[0][0] > self.hand_poses[1][0]:
                        # Swap hand poses in the list
                        tmp = self.hand_poses[0]
                        self.hand_poses[0] = self.hand_poses[1]
                        self.hand_poses[1] = tmp

                        if len(raw_robot_positions) > 1:
                            tmp_raw = raw_robot_positions[0]
                            raw_robot_positions[0] = raw_robot_positions[1]
                            raw_robot_positions[1] = tmp_raw

                        # Keep axes aligned with hand_poses
                        if len(self.wrist_axes) > 1:
                            tmp_axes = self.wrist_axes[0]
                            self.wrist_axes[0] = self.wrist_axes[1]
                            self.wrist_axes[1] = tmp_axes

                # Moving-average smoothing over last N frames (per hand index)
                for i in range(2):
                    if i < len(self.hand_poses):
                        self.pose_histories[i].append(self.hand_poses[i])
                        if i < len(self.wrist_axes):
                            self.axis_histories[i].append(self.wrist_axes[i])
                    else:
                        self.pose_histories[i].clear()
                        self.axis_histories[i].clear()

                smoothed_poses = []
                smoothed_axes = []
                for i in range(len(self.hand_poses)):
                    smoothed_poses.append(self._smooth_pose(self.pose_histories[i]))
                    smoothed_axes.append(self._smooth_axes(self.axis_histories[i]))

                self.hand_poses = smoothed_poses
                self.prev_hand_poses = self.hand_poses.copy()

                # Check for start and end gestures
                start_active = self.check_start_gesture()
                end_active = (not start_active) and self.check_end_gesture()

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
                            # Set the starting hand offsets in RAW space (pre-offset robot_pos).
                            # This avoids mixing smoothed/offset poses with raw landmark-derived positions.
                            if len(raw_robot_positions) >= 2:
                                self.start_left_offset = raw_robot_positions[0]
                                self.start_right_offset = raw_robot_positions[1]
                            print(f"Start gesture detected. Left offset: {self.start_left_offset}, Right offset: {self.start_right_offset}")

                            # Reset smoothing so we don't average pre-START poses into the new rebased frame.
                            for h in self.pose_histories:
                                h.clear()
                            for h in self.axis_histories:
                                h.clear()
                            if len(self.hand_poses) >= 2:
                                self.pose_histories[0].append(self._with_xyz(self.hand_poses[0], self.robot_left_offset))
                                self.pose_histories[1].append(self._with_xyz(self.hand_poses[1], self.robot_right_offset))
                        else:
                            self.callback_number = 0
                            self.episode_started = True
                elif end_active:
                    # Keep END text visible while gesture persists.
                    cv2.putText(annotated, 'END', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    if self.active_gesture != 'end':
                        self.active_gesture = 'end'
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
                for axes in smoothed_axes:
                    if axes is None:
                        continue
                    origin, x_axis, y_axis, z_axis = axes
                    for axis, color in zip([x_axis, y_axis, z_axis], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        end_pt = (int(origin[0] + axis[0] * scale), int(origin[1] + axis[1] * scale))
                        cv2.arrowedLine(annotated, origin, end_pt, color, 3, tipLength=0.2)

                # Logging print
                for idx, hand in enumerate(self.hand_poses):
                    hand_rpy = R.from_quat([hand[4], hand[5], hand[6], hand[3]]).as_euler('xyz', degrees=True)
                    output = "Left Hand:" if idx == 0 else "Right Hand:"
                    output += f"\n\tx={hand[0]:.2f} y={hand[1]:.2f} z={hand[2]:.2f}"
                    output += f"\n\tQw={hand[3]:.2f} Qx={hand[4]:.2f} Qy={hand[5]:.2f} Qz={hand[6]:.2f}"
                    # output += f"\n\tR={hand_rpy[0]:.1f} P={hand_rpy[1]:.1f} Y={hand_rpy[2]:.1f}\n"
                    output += f"\tThumb={hand[7]:.3f} Index={hand[8]:.3f} Pinky={hand[9]:.3f}"
                    print(output)

                if self.callback_number > 0:
                    print(f"Sending callback number: {self.callback_number}")

                # Send UDP message with hand pose data + callback number
                left_hand = self.hand_poses[0]
                right_hand = self.hand_poses[1] if len(self.hand_poses) > 1 else (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

                # On the START frame, force the reported xyz to equal the IsaacSim offsets exactly.
                # Orientation/fingers remain whatever is currently computed.
                if self.active_gesture == 'start' and self.callback_number == 1 and len(self.hand_poses) >= 2:
                    left_hand = self._with_xyz(left_hand, self.robot_left_offset)
                    right_hand = self._with_xyz(right_hand, self.robot_right_offset)
                pose_data = left_hand + right_hand + (float(self.callback_number),)

                msg = np.array(pose_data, dtype=np.float32).tobytes()
                self.sock.sendto(msg, (self.udp_ip, self.udp_port))

                self.callback_number = 0

            cv2.imshow('Hand Landmarker (World + Manual Draw)', annotated)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('-') or key == ord('_'):
                self.mirror = not self.mirror
                print(f"Mirror: {'ON' if self.mirror else 'OFF'}")
            elif key == ord('4') and primary_im_lm is not None:
                self.calibrate_step1(primary_im_lm, primary_hand_size_px, primary_rot_mat)
            elif key == ord('5') and primary_im_lm is not None and primary_hand_size_px is not None:
                self.calibrate_step2(primary_im_lm, primary_hand_size_px)
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
    ht = Handtracking(udp_ip=args.udp_ip, udp_port=args.udp_port)
    ht.tracking_loop()