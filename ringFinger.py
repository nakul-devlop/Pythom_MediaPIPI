import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# OpenCV to capture video
cap = cv2.VideoCapture(0)  # 0 is for the default webcam

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

    # If hand landmarks are found, draw only the ring finger landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Ring Finger landmarks: 13 -> base, 14 -> first joint, 15 -> second joint, 16 -> third joint, 17 -> tip
            ring_finger_landmarks = [13]
            
            # Draw connections between the ring finger landmarks
            for i in range(len(ring_finger_landmarks) - 1):
                start = ring_finger_landmarks[i]
                end = ring_finger_landmarks[i + 1]
                # Draw a line between consecutive ring finger landmarks
                cv2.line(frame,
                         (int(hand_landmarks.landmark[start].x * frame.shape[1]),
                          int(hand_landmarks.landmark[start].y * frame.shape[0])),
                         (int(hand_landmarks.landmark[end].x * frame.shape[1]),
                          int(hand_landmarks.landmark[end].y * frame.shape[0])),
                         (0, 0, 255), 2)  # red color, 2px thickness

            # Optionally, draw circles at the ring finger landmarks
            for i in ring_finger_landmarks:
                x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # blue circles for landmarks

    # Display the frame with the ring finger tracked
    cv2.imshow("Ring Finger with Ring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()