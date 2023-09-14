import torch
from PIL import Image
from pathlib import Path
import urllib.request

# Define the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')

# Load an image (replace with your image URL or local path)
image_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg"
urllib.request.urlretrieve(image_url, "transformer_image.jpg")  # Download the image
image = Image.open("transformer_image.jpg")

# Perform object detection on the image
results = model(image)

# Display the detected objects and their labels
results.show()

# Access the detected objects and their coordinates
for pred in results.pred[0]:
    class_id, conf, bbox = int(pred[5]), pred[4], pred[:4].tolist()
    label = model.names[class_id]
    print(f"Label: {label}, Confidence: {conf:.2f}, Bounding Box: {bbox}")
