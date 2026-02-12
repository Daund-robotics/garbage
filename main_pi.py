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
import RPi.GPIO as GPIO

# ================= GPIO & MOTORS =================
GPIO.setwarnings(False)

# GPIO Mode is set in automation_pre_test (BCM)
# We can re-assert it or trust it but automation_pre_test.py does: GPIO.setmode(GPIO.BCM)

# ---- DRIVE MOTORS ----
IN1, IN2 = 27, 17
IN3, IN4 = 22, 23
ENA, ENB = 12, 13

SPEED = 50
TURN_SPEED = 60

# ================= DRIVE FUNCTIONS =================
def init_drive_motors():
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
    global pwm_a, pwm_b
    pwm_a = GPIO.PWM(ENA, 1000)
    pwm_b = GPIO.PWM(ENB, 1000)
    pwm_a.start(SPEED)
    pwm_b.start(SPEED)

def stop_drive():
    GPIO.output([IN1, IN2, IN3, IN4], 0)

def forward():
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)
    GPIO.output(IN1,0); GPIO.output(IN2,1)
    GPIO.output(IN3,0); GPIO.output(IN4,1)

def backward():
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)
    GPIO.output(IN1,1); GPIO.output(IN2,0)
    GPIO.output(IN3,1); GPIO.output(IN4,0)

def left():
    pwm_a.ChangeDutyCycle(TURN_SPEED)
    pwm_b.ChangeDutyCycle(TURN_SPEED)
    GPIO.output(IN1,0); GPIO.output(IN2,1)
    GPIO.output(IN3,1); GPIO.output(IN4,0)

def right():
    pwm_a.ChangeDutyCycle(TURN_SPEED)
    pwm_b.ChangeDutyCycle(TURN_SPEED)
    GPIO.output(IN1,1); GPIO.output(IN2,0)
    GPIO.output(IN3,0); GPIO.output(IN4,1)


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

# Alignment Constants
TARGET_DISTANCE_MIN = 14
TARGET_DISTANCE_MAX = 20
CENTER_TOLERANCE = 50   # pixels

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

    try:
        init_drive_motors()
        print("Drive Motors Initialized")
    except Exception as e:
        print(f"Drive Motor Init Failed: {e}")

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
    
    # State tracking
    current_action = "Scanning"

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

        if garbage_detected_near:
             # We found pending garbage. Let's find the specific targetbox again to align
             # The existing logic above iterates all boxes. 
             # We need to Select the best target (closest/largest) from indices.
             
             target_box = None
             max_area = 0
             
             # Re-scan indices to find primary target
             for i in indices:
                idx = i if isinstance(i, (int, np.integer)) else i[0]
                box = boxes[idx]
                # Filter for garbage classes only
                cls_id = class_ids[idx]
                if cls_id < len(classNames) and classNames[cls_id] in garbage_map:
                    w, h = box[2], box[3]
                    area = w * h
                    if area > max_area:
                        max_area = area
                        target_box = box

             if target_box:
                x, y, w, h = target_box
                cx = x + w // 2
                img_center = INPUT_WIDTH // 2
                offset = cx - img_center
                
                dist_cm = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, min(w, h)) # Recalc just in case
                
                # Logic
                if abs(offset) > CENTER_TOLERANCE:
                    if offset > 0:
                        current_action = "Turning Right"
                        right()
                    else:
                        current_action = "Turning Left"
                        left()
                else:
                    # Centered
                    if dist_cm > TARGET_DISTANCE_MAX:
                        current_action = "Forward"
                        forward()
                    elif dist_cm < TARGET_DISTANCE_MIN:
                        current_action = "Backward"
                        backward()
                    else:
                        current_action = "Aligned"
                        stop_drive()
                        
                        if (time.time() - last_trigger_time > TRIGGER_COOLDOWN):
                            print("Aligned & In Range. Starting Automation...")
                            current_action = "Automation Running"
                            cv2.putText(img, current_action, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            cv2.imshow("Pi Garbage Detection", img)
                            cv2.waitKey(1)
                            
                            try:
                                automation_pre_test.automation_sequence()
                                last_trigger_time = time.time()
                            except Exception as e:
                                print(f"Automation Error: {e}")
                                
             else:
                current_action = "No Target"
                stop_drive()
        else:
             current_action = "Scanning"
             stop_drive()

        cv2.putText(img, f"Action: {current_action}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    stop_drive()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
