import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import smbus
import threading

# ================= GPIO & MOTORS =================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# ---- DRIVE MOTORS ----
IN1, IN2 = 27, 17
IN3, IN4 = 22, 23
ENA, ENB = 12, 13

# ---- LIFT MOTOR ----
LIFT_UP = 15
LIFT_DOWN = 14

# ---- LIMIT SWITCHES ----
L1 = 5  # Bottom
L2 = 6  # Top

# ---- SENSORS ----
WET_SENSOR_PIN = 20
METAL_SENSOR_PIN = 16

# ---- ULTRASONIC ----
# (Not used in this logic, relying on camera distance)

# ---- PCA9685 (SERVOS) ----
PCA_ADDR = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
SERVO_MIN = 102
SERVO_MAX = 512

# ---- CONSTANTS ----
TARGET_DISTANCE_MIN = 14
TARGET_DISTANCE_MAX = 20
CENTER_TOLERANCE = 50   # pixels
SPEED = 100
TURN_SPEED = 90

# Distance Estimation
KNOWN_WIDTH = 7.0  # cm (approx width of bottle/can)
FOCAL_LENGTH = 500 # Needs calibration

# ================= INIT =================
try:
    bus = smbus.SMBus(1)
except:
    print("SMBus not found (Mocking)")
    class MockSMBus:
        def write_byte_data(self, a,r,v): pass
        def read_byte_data(self, a,r): return 0
    bus = MockSMBus()

def init_gpio():
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, LIFT_UP, LIFT_DOWN], GPIO.OUT)
    GPIO.setup([L1, L2, WET_SENSOR_PIN, METAL_SENSOR_PIN], GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    global pwm_a, pwm_b
    pwm_a = GPIO.PWM(ENA, 1000)
    pwm_b = GPIO.PWM(ENB, 1000)
    pwm_a.start(SPEED)
    pwm_b.start(SPEED)
    
    init_pca()

# ================= DRIVE CONTROL =================
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

# ================= LIFT & SERVO CONTROL =================
def write_byte(reg, val):
    try: bus.write_byte_data(PCA_ADDR, reg, val)
    except: pass

def read_byte(reg):
    try: return bus.read_byte_data(PCA_ADDR, reg)
    except: return 0

def init_pca():
    write_byte(MODE1, 0x00)
    time.sleep(0.01)
    
    # Set frequency to 50Hz
    prescale = 121 # approx 50Hz
    oldmode = read_byte(MODE1)
    write_byte(MODE1, (oldmode & 0x7F) | 0x10)
    write_byte(PRESCALE, prescale)
    write_byte(MODE1, oldmode)
    time.sleep(0.005)
    write_byte(MODE1, oldmode | 0x80)

def set_pwm(ch, on, off):
    base = LED0_ON_L + 4 * ch
    write_byte(base, on & 0xFF)
    write_byte(base+1, on >> 8)
    write_byte(base+2, off & 0xFF)
    write_byte(base+3, off >> 8)

def move_servo(ch, angle):
    pulse = int(SERVO_MIN + (angle/180.0)*(SERVO_MAX-SERVO_MIN))
    set_pwm(ch, 0, pulse)
    time.sleep(0.2)

def lift_up_until_top():
    GPIO.output(LIFT_UP, 1); GPIO.output(LIFT_DOWN, 0)
    start = time.time()
    while GPIO.input(L2) == 1:
        if time.time() - start > 10: break
        time.sleep(0.05)
    GPIO.output(LIFT_UP, 0)

def lift_down_until_bottom(check_sensors=False):
    GPIO.output(LIFT_UP, 0); GPIO.output(LIFT_DOWN, 1)
    
    metal_confirmed = False
    wet_confirmed = False
    metal_start = None
    wet_start = None
    
    start = time.time()
    while GPIO.input(L1) == 1:
        if time.time() - start > 15: break
        
        if check_sensors:
            w = GPIO.input(WET_SENSOR_PIN)
            m = GPIO.input(METAL_SENSOR_PIN)
            
            # Logic: Sensors are typically active LOW (0 means detected)
            # automation_pre_test.py: WET=0 (detected), METAL=0 (detected)
            # Re-verifying logic from automation_pre_test.py:
            # "wet_raw == 1 and metal_raw == 0" -> Metal detected
            # "wet_raw == 0 and metal_raw == 1" -> Wet detected
            
            # Wait, let's look at automation_pre_test.py again:
            # if wet_raw == 1 and metal_raw == 0: ... metal_confirmed = True
            # if wet_raw == 0 and metal_raw == 1: ... wet_confirmed = True
            # This implies Active LOW for simple sensors or specific wiring?
            # Usually: 
            #   If sensor is LOW when active -> 0 is detected.
            #   If sensor is HIGH when active -> 1 is detected.
            # automation_pre_test seems to imply exclusive detection.
            
            if m == 0:  # Metal Detected
                 if metal_start is None: metal_start = time.time()
                 elif time.time() - metal_start > 2: metal_confirmed = True
            else: metal_start = None
            
            if w == 0: # Wet Detected
                 if wet_start is None: wet_start = time.time()
                 elif time.time() - wet_start > 2: wet_confirmed = True
            else: wet_start = None
            
        time.sleep(0.05)
    
    GPIO.output(LIFT_DOWN, 0)
    return metal_confirmed, wet_confirmed

# ================= AUTOMATION SEQUENCE =================
def run_automation():
    print("🤖 Starting Automation Sequence...")
    
    move_servo(1, 150) # Open Bucket
    lift_down_until_bottom()
    move_servo(1, 30)  # Close Bucket
    lift_up_until_top()
    
    time.sleep(2)
    move_servo(1, 150) # Drop
    
    metal, wet = lift_down_until_bottom(check_sensors=True)
    
    print(f"🧐 Analysis: Metal={metal}, Wet={wet}")
    
    if metal:
        print("-> Sorting Metal")
        move_servo(3, 20)
        move_servo(0, 140)
        time.sleep(3)
        move_servo(0, 30)
        move_servo(3, 90)
    elif wet:
        print("-> Sorting Wet")
        move_servo(3, 160)
        move_servo(0, 140)
        time.sleep(3)
        move_servo(0, 30)
        move_servo(3, 90)
    else:
        print("-> Sorting Dry/Other")
        move_servo(3, 90)
        move_servo(0, 140)
        time.sleep(3)
        move_servo(0, 30)
        move_servo(3, 90)
        
    lift_up_until_top()
    print("✅ Automation Complete")

# ================= YOLO & MAIN LOOP =================
def load_yolo():
    # Attempt to load YOLOv4-tiny if present, else warn or fallback?
    # Hardcoded to yolo/yolov4-tiny... as per request.
    net = cv2.dnn.readNet("yolo/yolov4-tiny.weights", "yolo/yolov4-tiny.cfg")
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def main():
    init_gpio()
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    
    print("🚀 System Ready. Press CTRL+C to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            class_ids = []
            confidences = []
            boxes = []
            
            # Parse detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            target_box = None
            max_area = 0
            
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    # Draw box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Find largest object (assumed to be the target garbage)
                    area = w * h
                    if area > max_area:
                        max_area = area
                        target_box = (x, y, w, h)

            # Control Logic
            if target_box:
                x, y, w, h = target_box
                cx = x + w // 2
                
                # Distance Calc
                dist_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / w
                
                # Label
                cv2.putText(frame, f"{int(dist_cm)}cm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                img_center = width // 2
                offset = cx - img_center
                
                status = "Idle"
                
                if abs(offset) > CENTER_TOLERANCE:
                    if offset > 0:
                        status = "Turning Right"
                        right()
                    else:
                        status = "Turning Left"
                        left()
                else:
                    # Centered, check distance
                    if dist_cm > TARGET_DISTANCE_MAX:
                        status = "Forward"
                        forward()
                    elif dist_cm < TARGET_DISTANCE_MIN:
                        status = "Backward"
                        backward()
                    else:
                        status = "Aligned! Starting Job..."
                        stop_drive()
                        run_automation()
                        # Back off a bit after job so we don't re-trigger immediately?
                        # Or sleep/wait
                        time.sleep(5)
                
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                stop_drive()
            
            cv2.imshow("Drive Motors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        stop_drive()
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
