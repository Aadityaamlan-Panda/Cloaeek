import cv2
import mediapipe as mp
import time
import pyttsx3
import random
import csv
import matplotlib.pyplot as plt
from collections import Counter, deque
from plyer import notification


# Initialize Mediapipe and Text-to-Speech
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Initialize Windows Notification
rate = engine.getProperty('rate')  # Get current speech rate
engine.setProperty('rate', rate - 40)
emotion_counter = 0  # Initialize the counter

# Define multiple messages for each emotion
emotion_messages = {
    "Focused": [
        "Great focus! Keep it up!",
        "You're doing fantastic. Stay focused!",
        "Focus is key. You're on the right track!",
        "Keep up the great work!",
        "You're in the zone, keep going!"
    ],
    "Tired": [
        "You look tired. Maybe it's time to stretch or grab some water?",
        "Rest is important. Take a short break!",
        "Feeling fatigued? A quick break could help!",
        "Maybe you need some rest. Recharge your energy!",
        "Take it easy, you've earned a break!"
    ],
    "Yawning": [
        "Feeling sleepy? You might need some rest.",
        "It's time to relax. Maybe take a nap?",
        "Yawning indicates rest is needed. Take a break!",
        "Feeling tired? Rest is essential.",
        "A good rest will recharge you."
    ],
    "Head Down": [
        "Seems like you need some rest. Consider taking a break.",
        "Your body needs a break. Rest up!",
        "Take a breather, you're doing great!",
        "Looks like you're feeling drained. Take a break.",
        "Rest your head and relax for a while."
    ],
    "Anxious": [
        "You seem anxious. Take a deep breath and relax.",
        "Try to calm down, everything will be okay.",
        "Take a moment to breathe deeply and relax.",
        "You're doing great. Calm your mind.",
        "Stay calm, take a deep breath."
    ],
    "Nervous": [
        "Are you feeling nervous? Stay calm and think things through.",
        "Nervousness is normal. Take a deep breath!",
        "You can do it! Stay confident and calm.",
        "Take a moment to calm your nerves.",
        "Stay calm, and everything will fall into place."
    ],
    "Surprised": [
        "Something surprised you? Stay composed.",
        "Stay calm, surprise is a part of life!",
        "Take a breath and process what's happening.",
        "Surprise can be overwhelming. Stay grounded.",
        "It’s okay to be surprised, just take it easy."
    ],
    "Thinking": [
        "Deep in thought? Take your time!",
        "You're in deep concentration. Keep it up!",
        "Great thought process, continue!",
        "Your thinking is sharp, keep going!",
        "Keep thinking, you're on the right path!"
    ],
    "Happy": [
        "You seem happy! Keep up the good vibes!",
        "Stay positive, you're doing great!",
        "Your happiness is contagious, keep it up!",
        "Great energy! Keep smiling!",
        "You're in a good mood, stay that way!"
    ],
        "Smiling": ["You look happy! Keep smiling!", "Your smile lights up the room!"],
    "Eye Fatigue": ["You might have eye fatigue. Blink more or take a break!"],
}

# Initialize CSV file for mood history
with open("mood_history.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Emotion"])

def log_emotion_to_csv(emotion):
    """Logs emotion with timestamp to CSV file."""
    with open("mood_history.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), emotion])

def show_notification(message):
    """Show system notification using Plyer."""
    notification.notify(
        title="Emotion Detected",
        message=message,
        timeout=5  # Notification duration in seconds
    )
def detect_emotion(face_landmarks, state):
    """
    Detects emotion and yawning based on face landmark positions.
    """
    left_eye = [33, 159]
    right_eye = [362, 386]
    mouth = [13, 14]
    nose = [1]
    lips = [61, 291]
    mouth_corners = [78, 308]

    if len(face_landmarks) > max(max(left_eye), max(right_eye), max(mouth), max(nose), max(lips)):
        left_eye_openness = abs(face_landmarks[left_eye[0]].y - face_landmarks[left_eye[1]].y)
        right_eye_openness = abs(face_landmarks[right_eye[0]].y - face_landmarks[right_eye[1]].y)
        mouth_openness = abs(face_landmarks[mouth[0]].y - face_landmarks[mouth[1]].y)
        lip_compression = abs(face_landmarks[lips[0]].x - face_landmarks[lips[1]].x)
        nose_position = face_landmarks[nose[0]].y
        mouth_corners_distance = abs(face_landmarks[mouth_corners[0]].x - face_landmarks[mouth_corners[1]].x)

        if 0.05 < mouth_corners_distance < 0.1 and 0.01 < mouth_openness < 0.03:
            return "Smiling"
        
        if left_eye_openness < 0.005 and right_eye_openness < 0.005:
            return "Eye Fatigue"

        if mouth_corners_distance > 0.1 and 0.02 < mouth_openness < 0.05:
            return "Happy"
        
        if face_landmarks[10].y > 0.7:  # Check if chin is low
            return "Head Down"

        if left_eye_openness < 0.015 and right_eye_openness < 0.015:
            return "Tired"
        
        if mouth_openness > 0.06:  # Yawning detection
            return "Yawning"
        
        if left_eye_openness > 0.04 and right_eye_openness > 0.04:  # Wide eyes
            return "Anxious"
        
        if lip_compression < 0.08 and mouth_openness < 0.02:  # Pressed lips
            return "Nervous"
        
        if mouth_openness > 0.03:
            return "Surprised"
        
        if abs(nose_position - 0.5) > 0.2:
            return "Thinking"

    return "Focused"

def notify_emotion(emotion,message):
    """Handles emotion notification."""
    if emotion in emotion_messages:
        show_notification(message)
        engine.say(message)
        engine.runAndWait()

# Camera and Mediapipe Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Graphing setup
plt.ion()
fig, ax = plt.subplots()
emotions_count = Counter()

with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, mp_hands.Hands() as hands:
    last_emotion_time = 0
    cooldown = 10  # Cooldown between messages (seconds)
    last_terminal_message_time = 0
    terminal_message_cooldown = 7  # seconds for terminal messages
    state = {"head_down_start": None}
    last_emotion = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        emotion = "No face detected"

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                emotion = detect_emotion(face_landmarks.landmark, state)
                if emotion == last_emotion:
                    emotion_counter += 1
                else:
                    emotion_counter = 1  # Reset counter when emotion changes

                last_emotion = emotion

                if time.time() - last_emotion_time > cooldown:
                    # Randomly choose a message for the detected emotion
                    message = random.choice(emotion_messages.get(emotion, ["No emotion detected."]))
                    print(f"Emotion: {emotion} - {message}")

                    if emotion_counter > 5:
                        try:
                            notify_emotion(emotion,message)
                        except Exception as e:
                            print(f"Error with pyttsx3: {e}")
                    
                    last_emotion_time = time.time()

                # Draw face landmarks
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Detect hand movements (e.g., face rubbing)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hand_near_face = any(
                    0.4 < landmark.x < 0.6 and 0.2 < landmark.y < 0.5 for landmark in hand_landmarks.landmark
                )
                if hand_near_face and time.time() - last_emotion_time > 5:
                    print("⚠️ You might be rubbing your face. Are you tired?")
                    engine.say("You might be rubbing your face. Are you tired?")
                    engine.runAndWait()
                    last_emotion_time = time.time()

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display emotion on screen
        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if time.time() - last_terminal_message_time > terminal_message_cooldown:
            if emotion == "Happy":
                print("😊 You seem happy! Keep up the good vibes!")
            elif emotion == "Tired":
                print("😔 Feeling sad? Remember, tough times never last!")
            elif emotion == "Focused":
                print("🎯 Stay focused, you're doing great!")
            last_terminal_message_time = time.time()

        emotions_count.update([emotion])
        ax.clear()
        ax.bar(emotions_count.keys(), emotions_count.values())
        ax.set_title("Mood Trends")
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.pause(0.01)
        
        cv2.imshow('Emotion-Aware Assistant', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
