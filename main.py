import time

import cv2
import mediapipe as mp
import csv
import numpy as np


# Create a VideoCapture object to capture video from the default camera (0)
cap = cv2.VideoCapture(0)

# Create a hands object
mp_hands = mp.solutions.hands

# Create a drawing object
mp_drawing = mp.solutions.drawing_utils
data = []

gestures = {'fist': 0, 'palm': 1, 'one': 2, 'two': 3, 'three': 4, 'pinch': 5 }  # Define the hand gestures and their labels

# Initialize variables for calculating framerate
frame_count = 0
start_time = 0
prev_time = 0


# Loop through frames from the video feed
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(frame_rgb)

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_data = []
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                l_x_min = 0
                l_y_min = 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                        l_x_min = landmark.x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                        l_y_min = landmark.y
                for landmark in hand_landmarks.landmark:
                    landmark_data.append(landmark.x-l_x_min)
                    landmark_data.append(landmark.y-l_y_min)
                    landmark_data.append(landmark.z)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Add the landmark data and label to the list
                data.append((landmark_data, gestures['pinch']))



        # Calculate and display the framerate
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time
        fps = 1 / elapsed_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        # If the user presses the 'q' key, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

with open('hand_gestures.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if csvfile.tell() == 0:  # Check if the file is empty
        writer.writerow(['landmark_{}_x, landmark_{}_y, landmark_{}_z'.format(i, i, i) for i in range(1, 22)] + ['label'])
    for row in data:
        writer.writerows([[str(num) for num in row[0]] + [str(row[1])]])