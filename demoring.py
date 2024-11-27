import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from objloader import *

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# OpenCV to capture video
cap = cv2.VideoCapture(0)

# Initialize OpenGL
glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(640, 480)
glutCreateWindow(b"3D Ring on Finger")
glEnable(GL_DEPTH_TEST)

# Load 3D model (make sure you have an obj loader class or method)
ring_model = ObjLoader("ring.obj")  # Load your 3D ring model

def setup_projection():
    gluPerspective(45, 1, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

def draw_ring(x, y, z, scale, rotation_angle):
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(rotation_angle, 0, 1, 0)  # Rotate around Y axis
    glScalef(scale, scale, scale)  # Scale the ring

    ring_model.render()  # Render the 3D ring

    glPopMatrix()

# Hand landmark processing
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
            # Ring Finger landmarks: 13 -> base, 14 -> first joint
            landmark_13 = hand_landmarks.landmark[13]  # base of ring finger
            landmark_14 = hand_landmarks.landmark[14]  # first joint of ring finger

            # Calculate the midpoint between landmarks 13 and 14
            x_mid = (landmark_13.x + landmark_14.x) / 2
            y_mid = (landmark_13.y + landmark_14.y) / 2

            # Convert the normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_mid_pixel = int(x_mid * w)
            y_mid_pixel = int(y_mid * h)

            # Calculate the angle of rotation
            delta_x = landmark_14.x - landmark_13.x
            delta_y = landmark_14.y - landmark_13.y
            angle = np.arctan2(delta_y, delta_x) * (180.0 / np.pi)  # angle in degrees

            # Scale the 3D ring model based on the finger length
            finger_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
            scale = finger_length * 0.3  # Adjust scale factor based on actual finger size

            # Map the 2D coordinates to 3D (you can adjust this as per your needs)
            # Assuming the Z-axis corresponds to the depth in camera space
            z = -5  # Set Z-value based on your scene requirements

            # Use OpenGL to draw the ring in the 3D space at the detected position
            draw_ring(x_mid_pixel / w, y_mid_pixel / h, z, scale, angle)

    # Show the camera feed
    cv2.imshow("Ring Finger with 3D Ring", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()