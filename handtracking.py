import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R


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
    LANDMARK_COLOR = (255, 255, 255)
    CONNECTION_COLOR = (0, 0, 0)
    THICKNESS = 2
    RADIUS = 4

    def __init__(self, udp_ip='127.0.0.1', udp_port=5005):
        self.hand_height_cm = 17.0  # measured real hand size cm (wrist to middle tip)
        self.ref_dist_1 = 15.0      # cm
        self.ref_dist_2 = 20.0      # cm
        self.ref_size_1 = None
        self.ref_size_2 = None
        self.ref_z_axis = None
        self.cm_per_px_1 = None
        self.cm_per_px_2 = None
        self.f_times_H = None
        self.calibrated = False

        self.callback_number = 0

        print(f"Press '1' to calibrate at {self.ref_dist_1}cm, then '2' at {self.ref_dist_2}cm from the camera (palm facing camera for orientation calibration).")
        print("Press ESC to quit.")

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


    def get_cm_per_px(self, z_cm):
        """
        Linearly interpolate cm-per-pixel at a given z (cm) using calibration data.
        """
        if self.cm_per_px_1 is None or self.cm_per_px_2 is None:
            return None
        # Linear interpolation between two calibration points
        z1, z2 = self.ref_dist_1, self.ref_dist_2
        c1, c2 = self.cm_per_px_1, self.cm_per_px_2
        if z2 == z1:
            return c1
        return c1 + (c2 - c1) * (z_cm - z1) / (z2 - z1)


    def real_world_distance(self, pt1_px, pt2_px, z_cm):
        """
        Estimate real-world distance (cm) between two points in image at given z (cm).
        pt1_px, pt2_px: (x, y) pixel coordinates
        z_cm: estimated distance from camera (cm)
        """
        cm_per_px = self.get_cm_per_px(z_cm)
        if cm_per_px is None:
            return None
        px_dist = np.linalg.norm(np.array(pt1_px) - np.array(pt2_px))
        return px_dist * cm_per_px


    def calibrate_step1(self, hand_size_px, z_axis):
        # Calculate cm-per-pixel at close calibration
        self.ref_size_1 = hand_size_px
        if hand_size_px > 0:
            self.cm_per_px_1 = self.hand_height_cm / hand_size_px
        print(f"Calibrated 1: {self.ref_size_1:.2f} px at {self.ref_dist_1}cm, cm/px={self.cm_per_px_1:.5f}")

        self.ref_z_axis = z_axis.copy() if np.linalg.norm(z_axis) > 0 else None
        if self.ref_z_axis is not None:
            print(f"Reference palm z_axis: {self.ref_z_axis}")


    def calibrate_step2(self, hand_size_px):
        # Calculate cm-per-pixel at far calibration
        self.ref_size_2 = hand_size_px
        if hand_size_px > 0:
            self.cm_per_px_2 = self.hand_height_cm / hand_size_px
        print(f"Calibrated 2: {self.ref_size_2:.2f} px at {self.ref_dist_2}cm, cm/px={self.cm_per_px_2:.5f}")

        self.f_times_H = (self.ref_size_1 * self.ref_dist_1 + self.ref_size_2 * self.ref_dist_2) / 2
        self.calibrated = True
        print(f"Calibration complete. f*H ≈ {self.f_times_H:.2f}")
        print(f"cm/px at {self.ref_dist_1}cm: {self.cm_per_px_1:.5f}")
        print(f"cm/px at {self.ref_dist_2}cm: {self.cm_per_px_2:.5f}")


    def tracking_loop(self):
        cap = cv2.VideoCapture(0)
        frame_timestamp_ms = 0
        cv2.namedWindow('Hand Landmarker (Pure Tasks + Manual Draw)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Landmarker (Pure Tasks + Manual Draw)', 1280, 720)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            frame_timestamp_ms += 33  # ~30 FPS

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            annotated = image.copy()

            if len(result.hand_landmarks) > 1:
                hand_distance = np.linalg.norm(np.array([result.hand_landmarks[0][0].x - result.hand_landmarks[1][0].x,
                                                          result.hand_landmarks[0][0].y - result.hand_landmarks[1][0].y]))
                if hand_distance < 0.25:
                    print(f"Warning: Hands are very close ({hand_distance:.3f}), removing second hand.")
                    result.hand_landmarks.pop(1)

            hand_poses = []
            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    h, w, _ = image.shape

                    # Draw connections
                    for start_idx, end_idx in self.HAND_CONNECTIONS:
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        start_px = (int(start.x * w), int(start.y * h))
                        end_px = (int(end.x * w), int(end.y * h))
                        cv2.line(annotated, start_px, end_px, self.CONNECTION_COLOR, self.THICKNESS)

                    # Draw landmarks
                    for lm in hand_landmarks:
                        px = (int(lm.x * w), int(lm.y * h))
                        cv2.circle(annotated, px, self.RADIUS, self.LANDMARK_COLOR, -1)

                    # Calibration and Z estimation
                    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
                    index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z])
                    middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y, hand_landmarks[9].z])
                    pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y, hand_landmarks[17].z])

                    wrist_px = np.array([wrist[0] * w, wrist[1] * h])
                    middle_px = np.array([middle_mcp[0] * w, middle_mcp[1] * h])
                    hand_size_px = np.linalg.norm(wrist_px - middle_px)

                    # Palm normal calculation
                    vec_index = index_mcp - wrist
                    vec_pinky = pinky_mcp - wrist
                    vec_middle = middle_mcp - wrist

                    z_axis = -vec_middle / np.linalg.norm(vec_middle)
                    y_axis = np.cross(vec_index, vec_pinky)
                    y_axis /= np.linalg.norm(y_axis)
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis /= np.linalg.norm(x_axis)

                    rot_mat = np.column_stack((x_axis, y_axis, z_axis))

                    # Apply reorientation matrix
                    reorient_mat = np.array(
                        [
                            [0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0],
                        ]
                    )
                    rot_mat = rot_mat @ reorient_mat

                    quat = R.from_matrix(rot_mat).as_quat()

                    # Draw axes
                    origin = (int(wrist[0] * w), int(wrist[1] * h))
                    scale = 80
                    end_x = (int(origin[0] + x_axis[0] * scale), int(origin[1] + x_axis[1] * scale))
                    cv2.arrowedLine(annotated, origin, end_x, (0, 0, 255), 3, tipLength=0.2)
                    end_y = (int(origin[0] + y_axis[0] * scale), int(origin[1] + y_axis[1] * scale))
                    cv2.arrowedLine(annotated, origin, end_y, (0, 255, 0), 3, tipLength=0.2)
                    end_z = (int(origin[0] + z_axis[0] * scale), int(origin[1] + z_axis[1] * scale))
                    cv2.arrowedLine(annotated, origin, end_z, (255, 0, 0), 3, tipLength=0.2)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('9'):
                        self.calibrate_step1(hand_size_px, z_axis)
                    elif key == ord('0'):
                        self.calibrate_step2(hand_size_px)
                    elif key == ord('1'):
                        self.callback_number = 1
                    elif key == ord('2'):
                        self.callback_number = 2
                    elif key == ord('3'):
                        self.callback_number = 3

                    if self.calibrated:
                        z = self.f_times_H / hand_size_px
                        cm_per_px = self.get_cm_per_px(z)
                        x = -cm_per_px * (wrist[0] * w - w / 2)
                        y = -cm_per_px * (wrist[1] * h - h / 2)
                        hand_poses.append((x, y, z, quat[3], quat[0], quat[1], quat[2]))  # (x, y, z, Qw, Qx, Qy, Qz)

            if hand_poses:
                # Separate left and right hands based on x position
                left_hand = hand_poses[0]
                right_hand = None
                if len(hand_poses) > 1:
                    if (hand_poses[0][0] < hand_poses[1][0]):
                        right_hand = hand_poses[1]
                    else:
                        left_hand = hand_poses[1]
                        right_hand = hand_poses[0]

                    # # Calulate the distance between the two hands in cm
                    # hand_distance_cm = np.linalg.norm(np.array(left_hand[:3]) - np.array(right_hand[:3]))
                    # if (hand_distance_cm < 5.0):
                    #     print(f"Warning: Hands are very close ({hand_distance_cm:.2f} cm), removing right hand.")
                    #     right_hand = None

                # Convert quaternion to RPY for debugging
                left_rpy = R.from_quat([left_hand[4], left_hand[5], left_hand[6], left_hand[3]]).as_euler('xyz', degrees=True)
                right_rpy = R.from_quat([right_hand[4], right_hand[5], right_hand[6], right_hand[3]]).as_euler('xyz', degrees=True) if right_hand is not None else None

                # Print hand poses
                output = ""
                if left_hand is not None:
                    output = f"Left Hand:\n\tx={left_hand[0]:.2f} y={left_hand[1]:.2f} z={left_hand[2]:.2f}"
                    output += f"\n\tQw={left_hand[3]:.2f} Qx={left_hand[4]:.2f} Qy={left_hand[5]:.2f} Qz={left_hand[6]:.2f}"
                    output += f"\n\tR={left_rpy[0]:.1f} P={left_rpy[1]:.1f} Y={left_rpy[2]:.1f}\n"
                if right_hand is not None:
                    output += f"Right Hand:\n\tx={right_hand[0]:.2f} y={right_hand[1]:.2f} z={right_hand[2]:.2f}"
                    output += f"\n\tQw={right_hand[3]:.2f} Qx={right_hand[4]:.2f} Qy={right_hand[5]:.2f} Qz={right_hand[6]:.2f}"
                    output += f"\n\tR={right_rpy[0]:.1f} P={right_rpy[1]:.1f} Y={right_rpy[2]:.1f}\n"
                print(output)

                if self.callback_number > 0:
                    print(f"Sending callback number: {self.callback_number}")

                if right_hand is None:
                    right_hand = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # default pose for missing hand

                # Send UDP packet with pose data as JSON
                pose_data = left_hand + right_hand + (float(self.callback_number),)
                msg = np.array(pose_data, dtype=np.float32).tobytes()
                self.sock.sendto(msg, (self.udp_ip, self.udp_port))
                self.callback_number = 0  # reset after sending

            cv2.imshow('Hand Landmarker (Pure Tasks + Manual Draw)', annotated)
            if cv2.waitKey(5) & 0xFF == 27:
                break

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