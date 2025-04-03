A real-time hand tracking and OCR system that allows users to draw shapes and write text in the air using their index finger. The system recognizes shapes (triangle, square, rectangle, and circle) and extracts handwritten text using Tesseract OCR. It also includes a gesture-based clearing function, where showing five fingers clears the screen.

Features
✅ Real-time Hand Tracking – Uses MediaPipe to detect and track hand movements.
✅ Shape Recognition – Identifies triangles, rectangles, squares, and circles.
✅ Text Recognition (OCR) – Extracts handwritten text using Tesseract OCR.
✅ Gesture-Based Clearing – Automatically clears the screen when five fingers are shown.
✅ Interactive Controls – Press S for shape recognition and T for text recognition.

Installation
Clone the Repository:
git clone https://github.com/yourusername/HandTracking-OCR.git
cd HandTracking-OCR

Install Dependencies:
pip install opencv-python mediapipe numpy pytesseract
Ensure Tesseract OCR is Installed:
Download and install Tesseract OCR and update its path in the script.

Usage
Run the script:
python hand_tracking_ocr.py

Move your index finger to draw.

Press S to recognize shapes.

Press T to recognize handwritten text.

Show five fingers to clear the screen.

Press Q to exit.
