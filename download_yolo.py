from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # This will automatically download the model
print("YOLOv8 model downloaded successfully!")
