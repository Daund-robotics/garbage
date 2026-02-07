import cv2
import cvzone
import math
from ultralytics import YOLO

# Initialize the webcam
# Use 0 for default webcam, 1 for external if 0 is built-in
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load the YOLO model
# Using yolov8s.pt (small) for better accuracy on rotated objects compared to nano.
model = YOLO("yolov8s.pt")

# Object classes for COCO dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Define garbage-relevant classes and their display names
garbage_map = {
    "bottle": "Plastic Bottle",
    "cup": "Metal Can",  # Assuming cans are often detected as cups in base COCO
    "wine glass": "Glass",
    "bowl": "Bowl",
    "banana": "Organic",
    "apple": "Organic",
    "sandwich": "Organic",
    "orange": "Organic",
    "broccoli": "Organic",
    "carrot": "Organic",
    "hot dog": "Organic",
    "pizza": "Organic",
    "donut": "Organic",
    "cake": "Organic",
}

# Distance Estimation Constants
# Adjust these based on your camera calibration
KNOWN_WIDTH = 7.0  # cm (average width of a bottle/can)
FOCAL_LENGTH = 700 # pixels (needs calibration, 600-800 is typical for 720p webcams)

def calculate_distance(focal_length, known_width, pixel_width):
    if pixel_width == 0:
        return 0
    return (known_width * focal_length) / pixel_width

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Enable Test Time Augmentation (augment=True) for better robustness at angles
    results = model(img, stream=True, augment=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter for garbage classes
            if currentClass in garbage_map and conf > 0.2:
                # Display Name
                displayName = garbage_map[currentClass]
                
                # Calculate Distance
                # Use min(w, h) to approximate the diameter (shorter side) of the object,
                # which allows distance estimation to work even if the object is horizontal.
                # Assuming KNOWN_WIDTH corresponds to the object's diameter/width.
                distance = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, min(w, h))
                
                # Determine Color
                # Green if < 20cm, Red otherwise
                if distance < 20: 
                    color = (0, 255, 0) # Green
                else:
                    color = (0, 0, 255) # Red

                # Draw Visuals
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color, colorC=color)
                
                # Display Text
                text = f'{displayName} {int(distance)}cm'
                cvzone.putTextRect(img, text, (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=5, colorR=color)

    cv2.imshow("Garbage Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
