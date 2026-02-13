import cv2
import socket
import struct
import pickle
import numpy as np
import threading
import time
import os
import urllib.request

# ================= USER CONFIGURATION =================
# 🔴 REPLACE THIS WITH THE IP ADDRESS OF YOUR RASPBERRY PI 🔴
RPI_IP = "192.168.1.100"  
# ======================================================

VIDEO_PORT = 5555
CMD_PORT = 5556

# --- YOLO CONFIGURATION ---
MODEL_TYPE = 'n' 
MODEL_FILE = "yolov8n.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

garbage_map = {
    "bottle": "Plastic Bottle", "cup": "Metal Can", "wine glass": "Glass", "bowl": "Bowl",
    "banana": "Organic", "apple": "Organic", "sandwich": "Organic", "orange": "Organic",
    "broccoli": "Organic", "carrot": "Organic", "hot dog": "Organic", "pizza": "Organic",
    "donut": "Organic", "cake": "Organic",
}

KNOWN_WIDTH = 7.0  # cm
FOCAL_LENGTH = 500 # Needs calibration

def calculate_distance(pixel_width):
    if pixel_width == 0: return 0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

# Global Frame
current_frame = None
lock = threading.Lock()
frame_ready = False

# Command Sender
cmd_socket = None

def maintain_command_connection():
    global cmd_socket
    while True:
        try:
            if cmd_socket is None:
                print(f"Connecting to Command Server at {RPI_IP}:{CMD_PORT}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((RPI_IP, CMD_PORT))
                cmd_socket = s
                print("✅ Connected to Command Server")
            time.sleep(1)
        except Exception as e:
            print(f"Command Connection Failed (Retrying): {e}")
            cmd_socket = None
            time.sleep(2)

def send_command(cmd):
    global cmd_socket
    if cmd_socket:
        try:
            cmd_socket.sendall(cmd.encode('utf-8'))
        except Exception as e:
            print(f"Send Error: {e}")
            cmd_socket = None # Force reconnect

def video_receiver():
    global current_frame, frame_ready
    
    while True:
        try:
            print(f"Connecting to Video Stream at {RPI_IP}:{VIDEO_PORT}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((RPI_IP, VIDEO_PORT))
            print("✅ Connected to Video Stream")
            
            data = b""
            payload_size = struct.calcsize("Q")
            
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4*1024)
                    if not packet: break
                    data += packet
                
                if len(data) < payload_size:
                    break # Connection lost
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                # Safety limit for buffer
                if msg_size > 10_000_000:
                    print("Image too large, skipping")
                    continue

                while len(data) < msg_size:
                    packet = client_socket.recv(4*1024)
                    if not packet: break
                    data += packet
                
                if len(data) < msg_size:
                    break

                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Decode
                frame_buffer = pickle.loads(frame_data)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                
                with lock:
                    current_frame = frame
                    frame_ready = True
                    
        except Exception as e:
            print(f"Video Stream Error: {e}")
            time.sleep(2)
        finally:
            client_socket.close()

def preprocess_image(img, input_size):
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (nw, nh))
    padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    pad_top = (input_size[1] - nh) // 2
    pad_left = (input_size[0] - nw) // 2
    padded_img[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized_img
    return padded_img, scale, (pad_top, pad_left)

def main():
    print(f"Starting Laptop Main Control - Target RPi: {RPI_IP}")
    
    # Init DNN
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        urllib.request.urlretrieve("https://github.com/yoobright/yolo-onnx/raw/main/yolov8n.onnx", MODEL_FILE)
        
    net = cv2.dnn.readNetFromONNX(MODEL_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Start Threads
    t_vid = threading.Thread(target=video_receiver)
    t_vid.daemon = True
    t_vid.start()
    
    t_cmd = threading.Thread(target=maintain_command_connection)
    t_cmd.daemon = True
    t_cmd.start()
    
    last_trigger_time = 0
    TRIGGER_COOLDOWN = 8 # Seconds (Give auto time to finish)
    
    CENTER_TOLERANCE = 50
    TARGET_MIN = 14
    TARGET_MAX = 20
    
    while True:
        with lock:
            if not frame_ready or current_frame is None:
                time.sleep(0.01)
                continue
            img = current_frame.copy()
        
        # YOLO Processing
        INPUT_WIDTH = INPUT_SIZE
        INPUT_HEIGHT = INPUT_SIZE
        padded_img, scale, (pad_top, pad_left) = preprocess_image(img, (INPUT_WIDTH, INPUT_HEIGHT))
        
        blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        outputs = np.transpose(outputs, (0, 2, 1))
        
        rows = outputs[0]
        boxes = []
        confidences = []
        class_ids = []
        
        for row in rows:
            classes_scores = row[4:]
            max_score_idx = np.argmax(classes_scores)
            max_score = classes_scores[max_score_idx]
            if max_score >= CONF_THRESHOLD:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                cx = (cx - pad_left) / scale
                cy = (cy - pad_top) / scale
                w /= scale
                h /= scale
                left = int(cx - 0.5 * w)
                top = int(cy - 0.5 * h)
                boxes.append([left, top, int(w), int(h)])
                confidences.append(float(max_score))
                class_ids.append(max_score_idx)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        
        target_box = None
        closest_dist = float('inf')
        
        if len(indices) > 0:
            for i in indices:
                idx = i if isinstance(i, (int, np.integer)) else i[0]
                box = boxes[idx]
                cls_id = class_ids[idx]
                
                if cls_id < len(classNames):
                    cls_name = classNames[cls_id]
                    if cls_name in garbage_map:
                        x, y, w, h = box
                        dist = calculate_distance(min(w, h))
                        
                        color = (0, 255, 0)
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(img, f"{garbage_map[cls_name]} {int(dist)}cm", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if dist < closest_dist:
                            closest_dist = dist
                            target_box = (x, y, w, h, dist)

        # Logic
        status = "Idle"
        if target_box:
            x, y, w, h, dist = target_box
            cx = x + w // 2
            img_center = img.shape[1] // 2
            offset = cx - img_center
            
            if abs(offset) > CENTER_TOLERANCE:
                if offset > 0:
                    status = "Turning Right"
                    send_command("RIGHT")
                else:
                    status = "Turning Left"
                    send_command("LEFT")
            else:
                if dist > TARGET_MAX:
                    status = "Forward"
                    send_command("FORWARD")
                elif dist < TARGET_MIN:
                    status = "Backward"
                    send_command("BACKWARD")
                else:
                    status = "Aligned"
                    if time.time() - last_trigger_time > TRIGGER_COOLDOWN:
                        status = "Starting Auto"
                        send_command("STOP")
                        time.sleep(0.5)
                        send_command("AUTO")
                        last_trigger_time = time.time()
                    else:
                         send_command("STOP")
        else:
            send_command("STOP")
            
        cv2.putText(img, f"CMD: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Laptop Control", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
