import cv2
from deepface import DeepFace
import time

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
camera = cv2.VideoCapture(0)

# FPS calculation
prev_time = 0

print("Starting Emotion Detection... Press SPACE to exit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:

        # Extract face region
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Analyze emotion
            result = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]

            label = f"{emotion} ({confidence:.1f}%)"

        except Exception as e:
            label = "Detecting..."

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Emotion text
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Window title
    cv2.imshow("Face Emotion Detection - Garvit", frame)

    # Press SPACE to exit
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

camera.release()
cv2.destroyAllWindows()