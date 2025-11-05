import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque


MOVEMENT_THRESHOLD = 0.05
CLICK_BEND_THRESHOLD = 0.07
COOLDOWN_MS = 250
SMOOTHING_FRAMES = 5
WINDOW_NAME = "🟡 PacMan Finger Controller"
WINDOW_POS = (20, 20)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

smooth_queue = deque(maxlen=SMOOTHING_FRAMES)

def normalized_to_np(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

def smooth_landmarks(lms):
    smooth_queue.append(lms)
    avg = np.mean(np.array(smooth_queue), axis=0)
    return avg

def direction_from_vector(v):
    mag = np.linalg.norm(v)
    if mag < MOVEMENT_THRESHOLD:
        return None

    ang = np.degrees(np.arctan2(v[1], v[0]))

    if -45 <= ang <= 45:
        return 'right'
    elif 45 < ang <= 135:
        return 'down'
    elif -135 <= ang < -45:
        return 'up'
    else:
        return 'left'


def is_thumbs_up(lm):
    """
    Detects a thumbs-up gesture:
    - Thumb tip above thumb base
    - Other fingers are folded
    """
    thumb_tip = lm[4]
    thumb_base = lm[2]
    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    
    thumb_up = thumb_tip[1] < thumb_base[1]  

    
    index_folded = index_tip[1] > lm[6][1]
    middle_folded = middle_tip[1] > lm[10][1]
    ring_folded = ring_tip[1] > lm[14][1]
    pinky_folded = pinky_tip[1] > lm[18][1]

    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, *WINDOW_POS)

    last_click_time = 0
    last_enter_time = 0
    current_dir = None
    print("🟡 Pac-Man Finger Controller Started 🟡")
    print("👉 Move index finger = Pac-Man moves | Bend = press key | 👍 = Enter")
    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera not found. Try index 1 or 2.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        direction = None
        click_trigger = False
        thumbs_up = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = np.array([normalized_to_np(l) for l in hand.landmark])

            lm = smooth_landmarks(lm)

            index_tip = lm[8][:2]
            index_base = lm[5][:2]
            wrist = lm[0][:2]
            palm_center = (index_base + wrist) / 2

            v = index_tip - palm_center
            direction = direction_from_vector(v)

            
            bend_dist = np.linalg.norm(index_tip - index_base)
            if bend_dist < CLICK_BEND_THRESHOLD:
                click_trigger = True

           
            thumbs_up = is_thumbs_up(lm)

            h, w = frame.shape[:2]
            p_tip = (int(index_tip[0] * w), int(index_tip[1] * h))
            p_palm = (int(palm_center[0] * w), int(palm_center[1] * h))

            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, p_tip, 10, (0, 0, 255), -1)
            cv2.circle(frame, p_palm, 8, (255, 255, 0), -1)
            cv2.line(frame, p_palm, p_tip, (0, 255, 255), 2)
            cv2.putText(frame, f"Dir: {direction or 'None'} | Click: {click_trigger} | 👍: {thumbs_up}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        try:
            if direction != current_dir:
                if current_dir:
                    pyautogui.keyUp(current_dir)
                if direction:
                    pyautogui.keyDown(direction)
                current_dir = direction
        except Exception as e:
            print(f"Key control error: {e}")

        now = time.time() * 1000

        
        if click_trigger and now - last_click_time > COOLDOWN_MS:
            if direction:
                pyautogui.press(direction)
                last_click_time = now

        
        if thumbs_up and now - last_enter_time > 1000:
            pyautogui.press('enter')
            print("✅ Thumbs-up detected → ENTER pressed")
            last_enter_time = now

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if current_dir:
        pyautogui.keyUp(current_dir)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

