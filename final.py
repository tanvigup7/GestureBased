import cv2
import mediapipe as mp
import threading
import time
from collections import Counter
import numpy as np
import sounddevice as sd
from pedalboard import Pedalboard, Chorus, Compressor, Delay, Gain, Reverb, Phaser
from pedalboard.io import AudioStream

# audio loop
fs = 44100
sd.default.samplerate = fs
recording_data = None
recording = False

def record_loop(duration=5):
    global recording_data
    print("[Loop] Recording...")
    recording_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("[Loop] Recording complete.")

def play_loop():
    global recording_data
    if recording_data is not None:
        print("[Loop] Playing...")
        sd.play(recording_data, fs)
        sd.wait()
        print("[Loop] Playback complete.")

# audio stream
def run_audio_stream():
    with AudioStream(
        input_device_name="MacBook Air Microphone",
        output_device_name="MacBook Air Speakers",
        allow_feedback=True
    ) as stream:
        stream.plugins = Pedalboard([
            Compressor(threshold_db=-50, ratio=25),
            Gain(gain_db=15),
            Chorus(rate_hz=0, depth=0.7, centre_delay_ms=0.0, feedback=0.1, mix=0.5),
            Phaser(),
        ])
        input("[Audio Stream] Press Enter to stop streaming...\n")

audio_thread = threading.Thread(target=run_audio_stream, daemon=True)
audio_thread.start()

# gesture
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

cap = cv2.VideoCapture(0)

w, h = 640, 480
tip_ids = [8, 12, 16, 20]
mid_ids = [6, 10, 14, 18]
gesture = ""
prev_gesture = None

def detect_fingers(landmarks):
    fingers = []
    for i in range(4):
        tip_y = landmarks[tip_ids[i]][2]
        mid_y = landmarks[mid_ids[i]][2]
        fingers.append(1 if tip_y < mid_y else 0)
    return fingers

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])

    if landmark_list:
        fingers = detect_fingers(landmark_list)
        up_count = sum(fingers)

        # define gestures
        if fingers == [0, 0, 0, 0]:
            gesture = "fist closed"
        elif fingers == [1, 0, 0, 0]:
            gesture = "1 finger up"
        elif fingers == [1, 1, 0, 0]:
            gesture = "2 fingers"
        elif fingers == [1, 1, 1, 1]:
            gesture = "open palm"
        else:
            gesture = ""

        if gesture:
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # gesture actions
        if gesture != prev_gesture:
            if gesture == "open palm":
                threading.Thread(target=record_loop, args=(5,)).start()
            elif gesture == "fist closed":
                threading.Thread(target=play_loop).start()
            elif gesture == "1 finger up":
                print("[Gesture] Reverb effect triggered (not yet implemented)")
            elif gesture == "2 fingers":
                print("[Gesture] Other effects triggered (not yet implemented)")
            prev_gesture = gesture

    cv2.imshow("Gesture Controlled Looper", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
