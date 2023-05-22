import cv2
import mediapipe as mp
import torch
from model_trainer import HandGestureClassifier
from collections import Counter

# Load the trained PyTorch model
net = HandGestureClassifier()
net.load_state_dict(torch.load('hand_gesture_classifier.pth'))
net.eval()

# Initialize the hand landmark detector
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

gestures = {}
gestures[(0,0)] = "no command"
gestures[(0,1)] = "turn on"
gestures[(1,0)] = "turn off"
gestures[(1,2)] = "turn off mode 1"
gestures[(1,3)] = "turn off mode 2"
gestures[(1,4)] = "turn off mode 3"
gestures[(0,2)] = "turn on mode 1"
gestures[(0,3)] = "turn on mode 2"
gestures[(0,4)] = "turn on mode 3"


last_poses = [None] * 30
first_pose = None
second_pose = None
count = 0
fin = False
fin2 = False
# Initialize the webcam stream
cap = cv2.VideoCapture(1)
mcv = 0
while True:
    # Capture a frame from the webcam stream
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB format and pass it to the hand landmark detector
    frame_n = frame
    h, w, c = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        results = hands.process(frame)

    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_data = []
            landmark_datas =[]
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
                landmark_data.append(landmark.x - l_x_min)
                landmark_data.append(landmark.y - l_y_min)
                landmark_data.append(landmark.z)
                landmark_datas = torch.tensor(landmark_data, dtype=torch.float32)
                landmark_datas = landmark_datas.unsqueeze(0)

             # Classify the hand gesture using the PyTorch model
            with torch.no_grad():
                outputs = net(landmark_datas)
                _, predicted = torch.max(outputs.data, 1)

            # Draw the hand landmark coordinates and predicted gesture label on the frame
            label = predicted.item()
            # track gestures for 30 frames
            last_poses[count] = predicted.item()
            if count < 29:
                count += 1
            else:
                fin = True
                count = 0
            # check the most frequent gesture and store the value
            if fin:
                counter = Counter(last_poses)
                most_common_value = counter.most_common(1)[0][0]
                print(most_common_value)
                mcv = most_common_value
                if first_pose is None:
                    first_pose = most_common_value
                    fin = False
                elif most_common_value != first_pose:
                    second_pose = most_common_value



            cv2.putText(frame_n, f'Gesture: {label}, command: {gestures.get((first_pose, second_pose))}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame_n, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame_n, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    else:
        first_pose = None
        second_pose = None
        fin = False

    # Display the annotated frame
    cv2.imshow('Hand Gesture Recognition', frame_n)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
