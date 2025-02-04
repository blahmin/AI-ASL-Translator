import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

model = load_model(r"SAVE PATH FOR keypoints_classification_model.h5")

num_to_label = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
                12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S", 18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

message = ""

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints, verbose=0)
            predicted_label = num_to_label[np.argmax(prediction)]
            cv2.putText(frame, f"Letter: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('z') or key == 16: 
                message += predicted_label
                print(f"Current message: {message}")

            if key == ord(' '):  
                message += " "
                print(f"Current message (with space): {message}")

    frame_height = frame.shape[0]
    message_box_height = 40
    frame_with_message = cv2.copyMakeBorder(frame, 0, message_box_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(frame_with_message, f"Message: {message}", (10, frame_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Keypoints Inference", frame_with_message)

    
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()

print(f"Final message: {message}")
