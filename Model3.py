# This code is using detectron2 to detect the hotspots , i found it better in decetecting the hotspots
# detectron2:
import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load an image
image_path = "transformer_image.jpg"  # Replace with the path to your transformer image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Failed to load the image.")
else:
    # Initialize Detectron2 model
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Perform object detection
    outputs = predictor(image)

    # Visualize the detected objects and their labels
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Display the result
    result_image = v.get_image()[:, :, ::-1]
    cv2.imshow("Transformer Image with Object Detection", result_image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
