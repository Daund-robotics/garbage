# ==================================
# hardware_test_panel.py
# ==================================

import RPi.GPIO as GPIO
import time
import Adafruit_PCA9685

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ----- Limit Switches -----
L1 = 5
L2 = 6

GPIO.setup(L1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(L2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ----- Sensors -----
METAL_SENSOR = 16
WET_SENSOR   = 20

GPIO.setup(METAL_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(WET_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ----- Lift Motor -----
LIFT_UP   = 15
LIFT_DOWN = 14

GPIO.setup(LIFT_UP, GPIO.OUT)
GPIO.setup(LIFT_DOWN, GPIO.OUT)

# ================= PCA9685 SETUP =================
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(50)

# Servo Channels
S0 = 0
S1 = 1
S3 = 3

# ================= SERVO FUNCTION =================
def set_servo(channel, angle):
    pulse = int(150 + (angle / 180.0) * 450)
    pwm.set_pwm(channel, 0, pulse)
    print(f"Servo Channel {channel} moved to {angle}°")

# ================= MOTOR FUNCTIONS =================
def lift_up():
    GPIO.output(LIFT_UP, True)
    GPIO.output(LIFT_DOWN, False)
    print("Lift Moving UP")

def lift_down():
    GPIO.output(LIFT_UP, False)
    GPIO.output(LIFT_DOWN, True)
    print("Lift Moving DOWN")

def lift_stop():
    GPIO.output(LIFT_UP, False)
    GPIO.output(LIFT_DOWN, False)
    print("Lift Stopped")

# ================= SENSOR STATUS FUNCTION =================
def read_sensors():
    metal = "DETECTED" if GPIO.input(METAL_SENSOR) == 0 else "NOT DETECTED"
    wet   = "DETECTED" if GPIO.input(WET_SENSOR) == 0 else "NOT DETECTED"

    print("---- Sensor Status ----")
    print("Metal Sensor :", metal)
    print("Wet Sensor   :", wet)
    print("-----------------------")

# ================= LIMIT SWITCH STATUS =================
def read_limits():
    l1 = "TRIGGERED" if GPIO.input(L1) == 0 else "NOT TRIGGERED"
    l2 = "TRIGGERED" if GPIO.input(L2) == 0 else "NOT TRIGGERED"

    print("---- Limit Switch Status ----")
    print("L1 (Bottom):", l1)
    print("L2 (Top)   :", l2)
    print("-----------------------------")

# ================= MAIN LOOP =================
print("===== Hardware Test Panel =====")
print("Commands:")
print("S0 10  → Move S0 to 10°")
print("S1 90  → Move S1 to 90°")
print("S3 60  → Move S3 to 60°")
print("LU     → Lift Up")
print("LD     → Lift Down")
print("STOP   → Stop Lift")
print("M      → Check Metal Sensor")
print("W      → Check Wet Sensor")
print("LIM    → Show Limit Switch Status")
print("EXIT   → Quit Program")
print("================================")

while True:

    cmd = input("Enter Command: ").strip().upper()

    # ---- Servo Commands ----
    if cmd.startswith("S0"):
        try:
            angle = int(cmd.split()[1])
            set_servo(S0, angle)
        except (IndexError, ValueError):
            print("Invalid format. Use S0 <angle>")

    elif cmd.startswith("S1"):
        try:
            angle = int(cmd.split()[1])
            set_servo(S1, angle)
        except (IndexError, ValueError):
             print("Invalid format. Use S1 <angle>")

    elif cmd.startswith("S3"):
        try:
            angle = int(cmd.split()[1])
            set_servo(S3, angle)
        except (IndexError, ValueError):
             print("Invalid format. Use S3 <angle>")

    # ---- Motor Commands ----
    elif cmd == "LU":
        lift_up()

    elif cmd == "LD":
         lift_down()

    elif cmd == "STOP":
        lift_stop()

    # ---- Sensors ----
    elif cmd == "M":
        if GPIO.input(METAL_SENSOR) == 0:
            print("Metal Sensor: DETECTED")
        else:
            print("Metal Sensor: NOT DETECTED")

    elif cmd == "W":
        if GPIO.input(WET_SENSOR) == 0:
            print("Wet Sensor: DETECTED")
        else:
            print("Wet Sensor: NOT DETECTED")

    elif cmd == "LIM":
        read_limits()

    elif cmd == "EXIT":
        lift_stop()
        GPIO.cleanup()
        print("Program Closed")
        break

    else:
        print("Invalid Command")
