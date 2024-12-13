import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detect_gesture(hand_landmarks):
    """Detect the gesture (rock, paper, or scissors) based on hand landmarks."""
    # Tip and MCP indices for thumb, index, middle, ring, and pinky
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcp = [2, 5, 9, 13, 17]

    # Get coordinates of landmarks
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    # Detect if fingers are open or closed
    finger_states = []
    for tip, mcp in zip(finger_tips, finger_mcp):
        finger_states.append(landmarks[tip][1] < landmarks[mcp][1])  # True if finger is open

    # Thumb check (special case since it moves horizontally)
    finger_states[0] = landmarks[4][0] > landmarks[3][0]

    # Map finger states to gestures
    if all(not state for state in finger_states):  # All fingers closed
        return "Rock"
    elif all(state for state in finger_states):  # All fingers open
        return "Paper"
    elif finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]:
        return "Scissors"  # Only index and middle fingers open
    else:
        return "Unknown"

# Setup directories for output
output_dir = "Output Images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(250, 250, 250), thickness=2, circle_radius=1),
                                         )
                # Detect gesture
                gesture = detect_gesture(hand_landmarks)
                print(f"Detected gesture: {gesture}")
                
                # Display gesture on image
                cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # Save our image    
        cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
