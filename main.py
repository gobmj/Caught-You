import cv2
import subprocess
from datetime import datetime
from PIL import Image

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

# Number of frames with face detected before locking
face_detected_count = 0
# Threshold to confirm detection (e.g., lock after 10 consecutive frames)
DETECTION_THRESHOLD = 10

def lock_screen():
    """Locks the screen on macOS."""
    subprocess.run(['pmset', 'displaysleepnow'])

def save_face(frame, face, margin=50):
    """Saves the detected face with extra margin as an image."""
    x, y, w, h = face

    # Calculate the extended area with a margin
    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, frame.shape[1])
    y_end = min(y + h + margin, frame.shape[0])

    # Crop the region including the face and the margin
    face_img_with_margin = frame[y_start:y_end, x_start:x_end]

    # Convert the image from BGR (OpenCV format) to RGB
    face_img_rgb = cv2.cvtColor(face_img_with_margin, cv2.COLOR_BGR2RGB)

    # Save the image using Pillow
    image = Image.fromarray(face_img_rgb)
    filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(filename)
    print(f"Face image with margin saved as {filename}")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # If faces are detected
    if len(faces) > 0:
        print(f"Detected {len(faces)} face(s)")

        # Increment the face detected count
        face_detected_count += 1

        # Save the face and lock the screen if detection threshold is reached
        if face_detected_count >= DETECTION_THRESHOLD:
            save_face(frame, faces[0])
            lock_screen()
            break
    else:
        # Reset the counter if no face is detected
        face_detected_count = 0

    # Display the camera feed (optional)
    cv2.imshow('Camera', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
