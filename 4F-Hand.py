import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time


MOVEMENT_THRESHOLD = 0.04     
CLICK_BEND_THRESHOLD = 0.06  
COOLDOWN_MS = 150             
WINDOW_NAME = "PacMan Finger Controller"
WINDOW_POS = (20, 20)       

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalized_to_np(landmark):
    """Convert a MediaPipe landmark to a numpy array (x, y, z)."""
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def direction_from_vector(v):
    """Determine direction (up, down, left, right) from vector."""
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


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, *WINDOW_POS)

    last_click_time = 0
    current_dir = None

    print("🟡 Pac-Man Finger Controller Started 🟡")
    print("👉 Move finger = move Pac-Man | Bend finger slightly = click (press key)")
    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera not detected. Try a different index (0→1→2).")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        direction = None
        click_trigger = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = [normalized_to_np(l) for l in hand.landmark]

            index_tip = lm[8][:2]
            index_base = lm[5][:2]
            wrist = lm[0][:2]
            palm_center = (index_base + wrist) / 2

            v = index_tip - palm_center
            direction = direction_from_vector(v)

            bend_dist = np.linalg.norm(index_tip - index_base)
            if bend_dist < CLICK_BEND_THRESHOLD:
                click_trigger = True

            h, w = frame.shape[:2]
            p_tip = (int(index_tip[0]*w), int(index_tip[1]*h))
            p_palm = (int(palm_center[0]*w), int(palm_center[1]*h))
            cv2.circle(frame, p_tip, 8, (0,0,255), -1)
            cv2.circle(frame, p_palm, 6, (255,0,0), -1)
            cv2.line(frame, p_palm, p_tip, (0,255,255), 2)
            cv2.putText(frame, f"Dir: {direction} Click: {click_trigger}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

      
        if direction != current_dir:
           
            if current_dir:
                pyautogui.keyUp(current_dir)
          
            if direction:
                pyautogui.keyDown(direction)
            current_dir = direction

       
        now = time.time() * 1000
        if click_trigger and now - last_click_time > COOLDOWN_MS:
            if direction:
                pyautogui.press(direction)
                last_click_time = now

        
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    
    if current_dir:
        pyautogui.keyUp(current_dir)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
