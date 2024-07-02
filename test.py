from ultralytics import YOLO
# model = YOLO("yolov8n.pt")  # load an official model
# Load a model
model = YOLO(r"path\to\best.pt")

# Run batched inference on a list of images
results = model([r"path\to\test\image.jpg"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    # result.save(filename=r"path\result.jpg")
