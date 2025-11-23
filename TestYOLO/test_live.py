# test_camera.py
# Requires: pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

# ------------------------------
# 1. Load model
# ------------------------------
model = YOLO("best.pt")  # replace with your model path

# ------------------------------
# 2. Set live camera source
# ------------------------------
camera_index = 1  # 0 = default webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Optional: set camera resolution (comment out if not needed)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ------------------------------
# 3. Process frames
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # YOLO inference
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

    # Annotation
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Live Camera", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# 4. Release
# ------------------------------
cap.release()
cv2.destroyAllWindows()
