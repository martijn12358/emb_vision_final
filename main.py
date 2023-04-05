import cv2
import mediapipe as mp
import csv


# Create a VideoCapture object to capture video from the default camera (0)
cap = cv2.VideoCapture(1)

# Create a hands object
mp_hands = mp.solutions.hands

# Create a drawing object
mp_drawing = mp.solutions.drawing_utils

landmark_data = []

# Initialize variables for calculating framerate
frame_count = 0
start_time = 0

# Loop through frames from the video feed
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(frame_rgb)

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_row = []
                for landmark in hand_landmarks.landmark:
                    landmark_row.append(landmark.x)
                    landmark_row.append(landmark.y)
                    landmark_row.append(landmark.z)
                landmark_data.append(landmark_row)

        # Display the resulting frame


        # Calculate and display the framerate
        if frame_count == 0:
            start_time = cv2.getTickCount()
        else:
            end_time = cv2.getTickCount()
            elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(fps)
        frame_count += 1

        cv2.imshow('frame', frame)

        # If the user presses the 'q' key, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

with open('hand_landmarks.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(landmark_data)