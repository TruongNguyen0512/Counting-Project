from ultralytics import YOLO # type: ignore
from PIL import Image  # type: ignore 
import cv2  # type: ignore
# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

results = model.predict(source="0",show=True) 