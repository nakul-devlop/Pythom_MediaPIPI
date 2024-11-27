import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Pygame for OpenGL window
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Set up OpenGL perspective
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Function to create a torus (ring) manually
def draw_ring(radius=0.3, tube_radius=0.1, slices=30, stacks=10):
    # Generate the torus geometry
    for i in range(slices):
        lat0 = np.pi * (-0.5 + float(i) / slices)
        lat1 = np.pi * (-0.5 + float(i + 1) / slices)
        
        glBegin(GL_QUAD_STRIP)
        
        for j in range(stacks + 1):
            lon = 2 * np.pi * float(j) / stacks
            x0 = (radius + tube_radius * np.cos(lat0)) * np.cos(lon)
            y0 = (radius + tube_radius * np.cos(lat0)) * np.sin(lon)
            z0 = tube_radius * np.sin(lat0)
            
            x1 = (radius + tube_radius * np.cos(lat1)) * np.cos(lon)
            y1 = (radius + tube_radius * np.cos(lat1)) * np.sin(lon)
            z1 = tube_radius * np.sin(lat1)
            
            glVertex3f(x0, y0, z0)
            glVertex3f(x1, y1, z1)
        
        glEnd()

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2 + (landmark2.z - landmark1.z) ** 2)

# Function to calculate the angle between three points (landmarks)
def calculate_angle(landmark1, landmark2, landmark3):
    vector1_x = landmark2.x - landmark1.x
    vector1_y = landmark2.y - landmark1.y
    vector2_x = landmark3.x - landmark2.x
    vector2_y = landmark3.y - landmark2.y
    dot_product = vector1_x * vector2_x + vector1_y * vector2_y
    mag1 = np.sqrt(vector1_x ** 2 + vector1_y ** 2)
    mag2 = np.sqrt(vector2_x ** 2 + vector2_y ** 2)
    cos_theta = dot_product / (mag1 * mag2)
    angle = np.arccos(cos_theta)  # Angle in radians
    return np.degrees(angle)  # Convert to degrees

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the ring finger landmarks: 13 -> base, 14 -> first joint, 15 -> second joint, 16 -> third joint
            landmark_13 = hand_landmarks.landmark[13]  # base of ring finger
            landmark_14 = hand_landmarks.landmark[14]  # first joint of ring finger
            landmark_15 = hand_landmarks.landmark[15]  # second joint of ring finger
            landmark_16 = hand_landmarks.landmark[16]  # third joint of ring finger

            # Calculate the distance between landmarks (for scaling the ring size)
            finger_distance = calculate_distance(landmark_13, landmark_14)
            scale_factor = finger_distance * 0.5  # Adjust scaling based on the distance

            # Calculate the angle of the finger using landmarks 13, 14, and 15
            angle = calculate_angle(landmark_13, landmark_14, landmark_15)

            # Use the ring finger's base (landmark_14) for positioning the ring
            x_pos = landmark_14.x * 2 - 1  # Normalize to OpenGL coordinate system (-1 to 1)
            y_pos = -(landmark_14.y * 2 - 1)  # Invert Y-axis to match OpenGL coordinates

            # Apply transformations to move, scale, and rotate the ring
            glPushMatrix()
            glTranslatef(x_pos, y_pos, -3)  # Move the ring to the finger position
            glRotatef(angle, 0, 0, 1)  # Rotate the ring to match the finger angle
            glScalef(scale_factor, scale_factor, scale_factor)  # Scale the ring based on finger size
            draw_ring()  # Draw the ring
            glPopMatrix()

    # Update the OpenGL window
    pygame.display.flip()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            quit()

    # Use OpenCV to display the original webcam feed
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()