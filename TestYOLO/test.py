# test_video.py
# Make sure you have ultralytics installed:
# pip install ultralytics

from ultralytics import YOLO
import cv2

# ------------------------------
# 1. Load your trained YOLOv8 model
# ------------------------------
model = YOLO("best.pt")  # replace with your model path

# ------------------------------
# 2. Set video source
# ------------------------------
video_path = "episode_000015.mp4"  # replace with your video path
save_path = "results/output.mp4"  # where to save the annotated video

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer to save results
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

# ------------------------------
# 3. Process frames
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

    # Annotate frame
    annotated_frame = results[0].plot()  # results[0] because predict returns a list

    # Display
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Save frame to output video
    out.write(annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# 4. Release resources
# ------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved to {save_path}")
