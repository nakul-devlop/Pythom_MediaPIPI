import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Load the ring image (ensure it's a PNG with transparency)
ring_image = cv2.imread('ring.png', cv2.IMREAD_UNCHANGED)  # with alpha channel

# OpenCV to capture video
cap = cv2.VideoCapture(0)  # 0 is for the default webcam

def calculate_distance(landmark1, landmark2):
    """ Calculate Euclidean distance between two landmarks """
    return np.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the results
    results = hands.process(rgb_frame)

    # If hand landmarks are found, draw the ring on the finger
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Ring Finger landmarks: 13 -> base, 14 -> first joint, 15 -> second joint
            landmark_13 = hand_landmarks.landmark[13]  # base of ring finger
            landmark_14 = hand_landmarks.landmark[14]  # first joint of ring finger

            # Calculate the midpoint between landmarks 13 and 14 (to place the ring)
            x_mid = (landmark_13.x + landmark_14.x) / 2
            y_mid = (landmark_13.y + landmark_14.y) / 2

            # Convert the normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_mid_pixel = int(x_mid * w)
            y_mid_pixel = int(y_mid * h)

            # Calculate the distance between landmarks 13 and 14 to estimate finger size
            finger_distance = calculate_distance(landmark_13, landmark_14)

            # Use the finger distance to scale the ring (adjust this scale factor as necessary)
            scale_factor = finger_distance * 300  # Multiply by a factor to adjust ring size
            ring_width = int(scale_factor)  # Set the width of the ring based on distance
            ring_height = int(scale_factor)  # Set the height of the ring based on distance

            # Resize the ring image to fit the finger
            ring_resized = cv2.resize(ring_image, (ring_width, ring_height))

            # Calculate the angle of the finger (between landmarks 12, 13, and 14)
            vector1_x = hand_landmarks.landmark[12].x - landmark_13.x
            vector1_y = hand_landmarks.landmark[12].y - landmark_13.y
            vector2_x = landmark_14.x - landmark_13.x
            vector2_y = landmark_14.y - landmark_13.y
            angle = np.arctan2(vector2_y, vector2_x) - np.arctan2(vector1_y, vector1_x)
            angle = np.degrees(angle)

            # Get the region of interest (ROI) where the ring will be placed
            y_offset = y_mid_pixel - ring_height // 2
            x_offset = x_mid_pixel - ring_width // 2

            # Ensure the ring fits within the frame boundaries
            if y_offset + ring_height > frame.shape[0]:
                y_offset = frame.shape[0] - ring_height
            if x_offset + ring_width > frame.shape[1]:
                x_offset = frame.shape[1] - ring_width

            # Rotate the ring to match the finger's angle
            M = cv2.getRotationMatrix2D((ring_resized.shape[1] // 2, ring_resized.shape[0] // 2), angle, 1)
            ring_rotated = cv2.warpAffine(ring_resized, M, (ring_resized.shape[1], ring_resized.shape[0]))

            # Add the rotated ring to the frame using alpha blending (transparency handling)
            for c in range(0, 3):  # Loop over the 3 color channels (BGR)
                frame[y_offset:y_offset + ring_height, x_offset:x_offset + ring_width, c] = \
                    frame[y_offset:y_offset + ring_height, x_offset:x_offset + ring_width, c] * \
                    (1 - ring_rotated[:, :, 3] / 255.0) + \
                    ring_rotated[:, :, c] * (ring_rotated[:, :, 3] / 255.0)

    # Display the frame with the ring placed on the finger
    cv2.imshow("Ring Finger with Ring Version 3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()