import cv2
import cvzone
import math
import numpy as np
import os
import urllib.request
import sys
import time
import threading
import automation_pre_test

# --- CONFIGURATION ---
# 'n' = Nano (Faster, Standard Accuracy)
# 's' = Small (Slower, Higher Accuracy)
# --- CONFIGURATION ---
# 'n' = Nano (Faster, Standard Accuracy)
# 's' = Small (Slower, Higher Accuracy)
MODEL_TYPE = 'n' 
# ---------------------

if MODEL_TYPE == 'n':
    MODEL_FILE = "yolov8n.onnx"
    # Using a reliable third-party source since official repo only hosts .pt
    MODEL_URL = "https://github.com/yoobright/yolo-onnx/raw/main/yolov8n.onnx"
    INPUT_SIZE = 640 # Reverted to 640
elif MODEL_TYPE == 's':
    MODEL_FILE = "yolov8s.onnx"
    # Using a reliable third-party source since official repo only hosts .pt
    MODEL_URL = "https://huggingface.co/pyronear/yolov8s/resolve/main/yolov8s.onnx"
    INPUT_SIZE = 640
else:
    print("Invalid MODEL_TYPE. Using nano.")
    MODEL_FILE = "yolov8n.onnx"
    MODEL_URL = "https://github.com/yoobright/yolo-onnx/raw/main/yolov8n.onnx"
    INPUT_SIZE = 640


INPUT_WIDTH = INPUT_SIZE
INPUT_HEIGHT = INPUT_SIZE
CONF_THRESHOLD = 0.20
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
        # User-Agent header sometimes helps with downloads
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def preprocess_image(img, input_size):
    """
    Resizes image to input_size with letterboxing to preserve aspect ratio.
    Returns: padded_img, scale, (pad_top, pad_left)
    """
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    # Resize
    resized_img = cv2.resize(img, (nw, nh))
    
    # Create blank canvas
    padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    
    # Center the resized image
    pad_top = (input_size[1] - nh) // 2
    pad_left = (input_size[0] - nw) // 2
    
    padded_img[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized_img
    
    return padded_img, scale, (pad_top, pad_left)

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ret, self.frame = self.capture.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        # Start the thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()
    
    def isOpened(self):
        return self.capture.isOpened()

def main():
    print(f"Starting Garbage Detection with {MODEL_FILE}...")
    
    # Initialize Automation
    try:
        automation_pre_test.init_pca()
        automation_pre_test.set_pwm_freq(50)
        time.sleep(0.5)
        automation_pre_test.set_defaults()
    except Exception as e:
        print(f"Automation Init Failed: {e}")

    # Check for model and download if missing
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found.")
        if not download_model(MODEL_URL, MODEL_FILE):
             print(f"Please manually download {MODEL_FILE} and place it in this directory.")
             input("Press Enter to exit...")
             return

    # Initialize OpenCV DNN Network
    net = cv2.dnn.readNetFromONNX(MODEL_FILE)
    
    # Use CPU by default to avoid CUDA errors without proper setup
    print("Using CPU for inference")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Initialize Webcam
    # Try indices 0 and 1 to find the USB camera
    cap = None
    for cam_index in [0, 1]:
        print(f"Trying to open camera index {cam_index}...")
        # Check if we can open it with standard VideoCapture first
        temp_cap = cv2.VideoCapture(cam_index)
        if temp_cap.isOpened():
            print(f"Successfully detected camera index {cam_index}")
            temp_cap.release()
            try:
                cap = ThreadedCamera(cam_index)
                if cap.isOpened():
                    print(f"Started ThreadedCamera on index {cam_index}")
                    break
            except Exception as e:
                print(f"Failed to start ThreadedCamera: {e}")
        else:
             temp_cap.release()
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam (tried index 0 and 1).")
        print("Please check your USB connection or try 'ls /dev/video*' in terminal.")
        return
        
    # cap.set(3, 640)  # Resolution 640x480 - Handled in ThreadedCamera
    # cap.set(4, 480)

    last_trigger_time = 0
    TRIGGER_COOLDOWN = 5

    while True:
        success, img = cap.read()
        if not success or img is None:
            # Wait a bit if frame not available
            time.sleep(0.01)
            continue
        
        start = time.time()

        # Preprocess Image with Letterboxing
        padded_img, scale, (pad_top, pad_left) = preprocess_image(img, (INPUT_WIDTH, INPUT_HEIGHT))
        
        blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
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

        # No need for x_factor/y_factor relative to original image here, 
        # since we will un-pad and un-scale later based on letterboxing

        for row in rows:
            classes_scores = row[4:]
            max_score_idx = np.argmax(classes_scores)
            max_score = classes_scores[max_score_idx]
            
            if max_score >= CONF_THRESHOLD:
                # Get box coordinates (cx, cy, w, h) relative to PADDED image
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Un-pad
                cx -= pad_left
                cy -= pad_top
                
                # Un-scale
                cx /= scale
                cy /= scale
                w /= scale
                h /= scale
                
                # Calculate top-left corner
                left = int(cx - 0.5 * w)
                top = int(cy - 0.5 * h)
                width = int(w)
                height = int(h)
                
                boxes.append([left, top, width, height])
                confidences.append(float(max_score))
                class_ids.append(max_score_idx)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        garbage_detected_near = False

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
                        garbage_detected_near = True
                    else:
                        color = (0, 0, 255) # Red

                    # Draw Visuals
                    cvzone.cornerRect(img, (top, left, width, height), l=9, rt=5, colorR=color, colorC=color)
                    
                    text = f'{displayName} {int(distance)}cm'
                    cvzone.putTextRect(img, text, (max(0, left), max(35, top)), scale=1.5, thickness=2, offset=5, colorR=color)

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(img, f"FPS: {int(fps)} Model: {MODEL_TYPE.upper()} Res: {INPUT_SIZE}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Pi Garbage Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if garbage_detected_near and (time.time() - last_trigger_time > TRIGGER_COOLDOWN):
            print("Garbage < 20cm. Starting Automation...")
            try:
                automation_pre_test.automation_sequence()
                last_trigger_time = time.time()
            except Exception as e:
                print(f"Automation Error: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
