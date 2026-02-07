import cv2
import cvzone
import math
from ultralytics import YOLO
import time

# Initialize the webcam
# Pi Camera often uses index 0. If using USB Cam, might be 0 or 1.
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Lower resolution for Pi (640x480) to improve FPS
cap.set(4, 480)

# Load the Optimized YOLO model (NCNN)
# INFO: You MUST run 'yolo export model=yolov8s.pt format=ncnn' first!
# If that folder doesn't exist, change this back to 'yolov8s.pt' but expect 0.5 FPS.
try:
    model = YOLO("yolov8s_ncnn_model") 
    print("Loaded NCNN optimized model.")
except:
    print("NCNN model not found. Loading standard .pt model (Slow!)")
    print("Run 'yolo export model=yolov8s.pt format=ncnn' to fix this.")
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
    "cup": "Metal Can",
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
KNOWN_WIDTH = 7.0  # cm
FOCAL_LENGTH = 500 # Adjusted estimate for lower resolution (needs calibration)

def calculate_distance(focal_length, known_width, pixel_width):
    if pixel_width == 0:
        return 0
    return (known_width * focal_length) / pixel_width

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Measure FPS
    start = time.time()

    # Disable augmentation on Pi for speed
    results = model(img, stream=True, augment=False)

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
            if currentClass in garbage_map and conf > 0.25:
                # Display Name
                displayName = garbage_map[currentClass]
                
                # Calculate Distance using min dimension (diameter)
                distance = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, min(w, h))
                
                # Determine Color
                if distance < 20: 
                    color = (0, 255, 0) # Green
                else:
                    color = (0, 0, 255) # Red

                # Draw Visuals (Simplified for performance check)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color, colorC=color)
                cvzone.putTextRect(img, f'{displayName} {int(distance)}cm', (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=5, colorR=color)

    # FPS Display
    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(img, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Pi Garbage Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
