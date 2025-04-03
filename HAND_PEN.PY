import cv2
import mediapipe as mp
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class HandPen:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_points = []  
        self.shape = ""  
        self.recognized_text = ""

    def track_hand(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                index_tip = hand_landmarks.landmark[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                self.drawing_points.append((x, y))

                for i in range(1, len(self.drawing_points)):
                    cv2.line(frame, self.drawing_points[i - 1], self.drawing_points[i], (255, 0, 255), 7)

                
                if self.count_fingers(hand_landmarks) == 5:
                    self.clear_screen()

       
        cv2.putText(frame, f"Shape: {self.shape}", (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Text: {self.recognized_text}", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def count_fingers(self, hand_landmarks):
        """Counts raised fingers based on landmark positions."""
        tips = [8, 12, 16, 20]  
        thumb_tip = 4
        fingers = 0

        for tip in tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers += 1

        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            fingers += 1

        return fingers

    def recognize_shape(self):
        """Detects basic shapes (triangle, circle, square, rectangle)."""
        if len(self.drawing_points) < 20:
            print("⚠ Not enough points to detect a shape!")
            return

        canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255  
        for i in range(1, len(self.drawing_points)):
            cv2.line(canvas, self.drawing_points[i - 1], self.drawing_points[i], (0, 0, 0), 8)

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"🔍 Found {len(contours)} contours")  

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:  
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            sides = len(approx)

            print(f"🔹 Detected {sides} sides")  

            if sides == 3:
                self.shape = "Triangle"
            elif sides == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    self.shape = "Square"
                else:
                    self.shape = "Rectangle"
            elif sides > 4:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circle_area = np.pi * (radius ** 2)
                fill_ratio = area / circle_area  
                if fill_ratio > 0.75:
                    self.shape = "Circle"
                else:
                    self.shape = "Polygon"

        print(f"✅ Recognized Shape: {self.shape}")

    def recognize_text(self):
        """Extracts text from the drawing using OCR."""
        if len(self.drawing_points) < 20:
            print("⚠ Not enough points to recognize text!")
            return

        canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255  
        for i in range(1, len(self.drawing_points)):
            cv2.line(canvas, self.drawing_points[i - 1], self.drawing_points[i], (0, 0, 0), 8)

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        self.recognized_text = pytesseract.image_to_string(blurred, config="--psm 6").strip()

        print(f"✅ Recognized Text: {self.recognized_text}")

    def clear_screen(self):
        """Clears the screen for a new drawing."""
        self.drawing_points.clear()
        self.shape = ""
        self.recognized_text = ""
        print("🧹 Screen Cleared!")


hand_pen = HandPen()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  
    tracked_frame = hand_pen.track_hand(frame)

    cv2.imshow("Hand Drawing with Shape & Text Recognition", tracked_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("🚀 Shape recognition triggered!")  
        hand_pen.recognize_shape()  
    if key == ord('t'):
        print("🔠 Text recognition triggered!")  
        hand_pen.recognize_text()  
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
