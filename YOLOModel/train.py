# Install YOLOv8 if you haven't already
# pip install ultralytics

from ultralytics import YOLO

# Step 1: Create a YOLO model (you can use 'yolov8n.pt' for nano, 'yolov8s.pt' for small, etc.)
model = YOLO("yolov8n.yaml")  # the architecture config, no pre-trained weights

# Step 2: Train the model
# You need a YAML file describing your dataset, e.g., dataset.yaml:
# train: /path/to/dataset/images/train
# val: /path/to/dataset/images/val
# nc: number_of_classes
# names: ['class1', 'class2', ...]

model.train(
    data="dataset.yaml",   # path to your dataset YAML
    epochs=100,             # number of epochs
    imgsz=(640,480),             # image size
    batch=4,              # batch size
    device='cpu'               # GPU device (0 for first GPU, 'cpu' for CPU)
)

# Step 3 (Optional): Evaluate the trained model
metrics = model.val()
print(metrics)

# Step 4 (Optional): Export to ONNX, CoreML, etc.
model.export(format="onnx")