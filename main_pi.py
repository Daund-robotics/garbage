import cv2
import cvzone
import math
import numpy as np
import os
import urllib.request
import sys
import time

# --- CONFIGURATION ---
# 'n' = Nano (Faster, Standard Accuracy)
# 's' = Small (Slower, Higher Accuracy)
MODEL_TYPE = 'n' 
# ---------------------

if MODEL_TYPE == 'n':
    MODEL_FILE = "yolov8n.onnx"
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
    INPUT_SIZE = 640 # Standard size for good accuracy
elif MODEL_TYPE == 's':
    MODEL_FILE = "yolov8s.onnx"
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.onnx"
    INPUT_SIZE = 640 
else:
    print("Invalid MODEL_TYPE. Using nano.")
    MODEL_FILE = "yolov8n.onnx"
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
    INPUT_SIZE = 640


INPUT_WIDTH = INPUT_SIZE
INPUT_HEIGHT = INPUT_SIZE
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

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
FOCAL_LENGTH = 500 # Adjusted for lower resolution (needs recalibration)

def calculate_distance(focal_length, known_width, pixel_width):
    if pixel_width == 0:
        return 0
    return (known_width * focal_length) / pixel_width

def download_model(url, path):
    print(f"Downloading {path} from {url}...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def main():
    print(f"Starting Garbage Detection with {MODEL_FILE}...")
    
    # Check for model and download if missing
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found.")
        if not download_model(MODEL_URL, MODEL_FILE):
             print(f"Please manually download {MODEL_FILE} and place it in this directory.")
             input("Press Enter to exit...")
             return

    # Initialize OpenCV DNN Network
    net = cv2.dnn.readNetFromONNX(MODEL_FILE)
    
    # Try to use CUDA if available, else CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except:
        # On RPi, try to use OpenCL (optional) but CPU is standard
        # simple CPU fallback
        # print("CUDA not available, utilizing CPU.") 
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Resolution 640x480
    cap.set(4, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, img = cap.read()
        if not success:
            break
        
        start = time.time()

        # Preprocess Image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Forward Pass
        outputs = net.forward()
        
        # Output shape is (1, 84, 8400) -> Transpose to (1, 8400, 84)
        outputs = np.transpose(outputs, (0, 2, 1)) 
        
        # Extract rows
        rows = outputs[0]
        
        boxes = []
        confidences = []
        class_ids = []

        image_h, image_w, _ = img.shape
        x_factor = image_w / INPUT_WIDTH
        y_factor = image_h / INPUT_HEIGHT

        for row in rows:
            classes_scores = row[4:]
            max_score_idx = np.argmax(classes_scores)
            max_score = classes_scores[max_score_idx]
            
            if max_score >= CONF_THRESHOLD:
                # Get box coordinates (cx, cy, w, h)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Scale back to original image
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                confidences.append(float(max_score))
                class_ids.append(max_score_idx)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        for i in indices:
            # Depending on opencv version, i might be a list or int
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            
            box = boxes[idx]
            left, top, width, height = box[0], box[1], box[2], box[3]
            conf = confidences[idx]
            cls_id = class_ids[idx]
            
            if cls_id < len(classNames):
                currentClass = classNames[cls_id]
                
                # Filter for garbage classes
                if currentClass in garbage_map:
                    displayName = garbage_map[currentClass]
                    
                    # Distance Calculation
                    distance = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, min(width, height))
                    
                    # Color Logic
                    if distance < 20: 
                        color = (0, 255, 0) # Green
                    else:
                        color = (0, 0, 255) # Red

                    # Draw Visuals
                    cvzone.cornerRect(img, (left, top, width, height), l=9, rt=5, colorR=color, colorC=color)
                    
                    text = f'{displayName} {int(distance)}cm'
                    cvzone.putTextRect(img, text, (max(0, left), max(35, top)), scale=1.5, thickness=2, offset=5, colorR=color)

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(img, f"FPS: {int(fps)} Model: {MODEL_TYPE.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Pi Garbage Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
