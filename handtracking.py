import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

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

    def __init__(self):
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

        print(f"Press '1' to calibrate at {self.ref_dist_1}cm, then '2' at {self.ref_dist_2}cm from the camera (palm facing camera for orientation calibration).")
        print("Press ESC to quit.")

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


    def vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])


    def normalized_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


    def rot_matrix_to_rpy(self, R):
        yaw = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        roll = np.arctan2(R[2,1], R[2,2])
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


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
                    wrist = hand_landmarks[0]
                    pt_5 = hand_landmarks[5]
                    pt_9 = hand_landmarks[9]
                    pt_17 = hand_landmarks[17]

                    wrist_px = np.array([wrist.x * w, wrist.y * h])
                    middle_px = np.array([pt_9.x * w, pt_9.y * h])
                    hand_size_px = np.linalg.norm(wrist_px - middle_px)

                    # Palm normal calculation
                    vec_05 = self.vector(pt_5, wrist)
                    vec_017 = self.vector(pt_17, wrist)
                    vec_09 = self.vector(pt_9, wrist)

                    x_axis = self.normalized_vector(vec_09)
                    z_axis = self.normalized_vector(np.cross(vec_05, vec_017))
                    y_axis = self.normalized_vector(np.cross(z_axis, x_axis))

                    R = np.column_stack((x_axis, y_axis, z_axis))
                    roll, pitch, yaw = self.rot_matrix_to_rpy(R)

                    # Draw axes
                    origin = (int(wrist.x * w), int(wrist.y * h))
                    scale = 80
                    end_x = (int(origin[0] + x_axis[0] * scale), int(origin[1] + x_axis[1] * scale))
                    cv2.arrowedLine(annotated, origin, end_x, (0, 0, 255), 3, tipLength=0.2)
                    end_y = (int(origin[0] + y_axis[0] * scale), int(origin[1] + y_axis[1] * scale))
                    cv2.arrowedLine(annotated, origin, end_y, (0, 255, 0), 3, tipLength=0.2)
                    end_z = (int(origin[0] + z_axis[0] * scale), int(origin[1] + z_axis[1] * scale))
                    cv2.arrowedLine(annotated, origin, end_z, (255, 0, 0), 3, tipLength=0.2)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('1'):
                        self.calibrate_step1(hand_size_px, z_axis)
                    if key == ord('2'):
                        self.calibrate_step2(hand_size_px)

                    if self.calibrated:
                        z = self.f_times_H / hand_size_px
                        cm_per_px = self.get_cm_per_px(z)
                        x = -cm_per_px * (wrist.x * w - w / 2)
                        y = -cm_per_px * (wrist.y * h - h / 2)
                        hand_poses.append((x, y, z, roll, pitch, yaw))

            if hand_poses:
                left_hand = hand_poses[0]
                right_hand = None
                if len(hand_poses) > 1:
                    if (hand_poses[0][0] < hand_poses[1][0]):
                        right_hand = hand_poses[1]
                    else:
                        left_hand = hand_poses[1]
                        right_hand = hand_poses[0]

                output = ""
                if (left_hand is not None):
                    output = f"Left Hand: x={left_hand[0]:.2f} y={left_hand[1]:.2f} z={left_hand[2]:.2f} R={left_hand[3]:.1f} P={left_hand[4]:.1f} Y={left_hand[5]:.1f}\t\t|\t\t"
                if (right_hand is not None):
                    output += f"Right Hand: x={right_hand[0]:.2f} y={right_hand[1]:.2f} z={right_hand[2]:.2f} R={right_hand[3]:.1f} P={right_hand[4]:.1f} Y={right_hand[5]:.1f}"
                print(output)

            cv2.imshow('Hand Landmarker (Pure Tasks + Manual Draw)', annotated)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("Done.")

if __name__ == "__main__":
    ht = Handtracking()
    ht.tracking_loop()