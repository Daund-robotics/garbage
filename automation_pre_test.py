# ================================
# automation_pre_test.py
# ================================

import RPi.GPIO as GPIO
import time
import Adafruit_PCA9685

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ----- Limit Switches -----
L1 = 5   # Bottom
L2 = 6   # Top

GPIO.setup(L1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(L2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ----- Sensors -----
METAL_SENSOR = 16
WET_SENSOR   = 20

GPIO.setup(METAL_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(WET_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ----- Lift Motor (L298N) -----
LIFT_UP   = 15
LIFT_DOWN = 14

GPIO.setup(LIFT_UP, GPIO.OUT)
GPIO.setup(LIFT_DOWN, GPIO.OUT)

# ================= PCA9685 SETUP =================
# Try enabling I2C bus 1 explicitly if needed, but library defaults are usually good
try:
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(50)  # 50Hz for servos
except Exception as e:
    print(f"Error initializing PCA9685: {e}")
    print("Ensure 'Adafruit_PCA9685' is installed: pip install adafruit-pca9685")

# Servo Channels
S0 = 0  # Garbage Dumper
S1 = 1  # Pick Bucket
S3 = 3  # Sorting Gate

# ================= SERVO FUNCTION =================
def set_servo(channel, angle):
    # Map 0-180 to 150-600 pulse width (approximate standard range)
    pulse = int(150 + (angle / 180.0) * 450)
    try:
        pwm.set_pwm(channel, 0, pulse)
    except:
        pass # Handle potential I2C errors gracefully

# ================= MOTOR FUNCTIONS =================
def lift_up():
    GPIO.output(LIFT_UP, True)
    GPIO.output(LIFT_DOWN, False)

def lift_down():
    GPIO.output(LIFT_UP, False)
    GPIO.output(LIFT_DOWN, True)

def lift_stop():
    GPIO.output(LIFT_UP, False)
    GPIO.output(LIFT_DOWN, False)

# ================= DEFAULT POSITION =================
def set_default_positions():
    set_servo(S0, 0)
    set_servo(S1, 150)
    set_servo(S3, 0)
    print("✅ Default position is set")

# ================= INITIALIZATION =================
if __name__ == "__main__":
    try:
        set_default_positions()

        # ================= MOVE LIFT UP UNTIL L2 =================
        print("Checking L2 position...")
        if GPIO.input(L2) == 1: # Not triggered (HIGH due to Pull-Up)
            print("Moving Lift UP...")
            while GPIO.input(L2) == 1:
                lift_up()
                time.sleep(0.01)
        lift_stop()
        print("✅ Homing Complete (L2 Triggered)")
        set_default_positions()

        print("System Ready. Type 'N' to start sequence.")

        # ================= MAIN LOOP =================
        while True:

            cmd = input("Enter Command: ")

            if cmd.upper() == "N":

                print("🚀 Starting Automation Sequence")

                # Step 1
                set_servo(S1, 150)
                time.sleep(0.3)

                # Step 2 - Lift Down until L1
                print("⬇️ Lift DOWN to L1...")
                while GPIO.input(L1) == 1:
                    lift_down()
                    time.sleep(0.001)
                lift_stop()

                set_servo(S1, 30)
                time.sleep(0.5)

                # Step 3 - Lift Up until L2
                print("⬆️ Lift UP to L2...")
                while GPIO.input(L2) == 1:
                    lift_up()
                    time.sleep(0.001)
                lift_stop()

                print("⏳ Waiting 4s...")
                time.sleep(4)

                # Step 4
                set_servo(S1, 150)

                metal_detected = False
                wet_detected = False

                # Step 5 - Lift Down and check sensors simultaneously
                print("⬇️ Lift DOWN scanning sensors...")
                while GPIO.input(L1) == 1:
                    lift_down()

                    if GPIO.input(METAL_SENSOR) == 0:
                        metal_detected = True
                        print("🧲 Metal Detected!")

                    if GPIO.input(WET_SENSOR) == 0:
                        wet_detected = True
                        print("💧 Wet Detected!")
                    
                    time.sleep(0.001)

                lift_stop()

                # Step 6 - Sorting
                if metal_detected:
                    print("➡️ Metal Detected -> Metal Bin")
                    set_servo(S3, 0)
                    time.sleep(0.1)
                    set_servo(S0, 140)
                    time.sleep(3)
                    set_servo(S0, 0)

                elif wet_detected:
                    print("➡️ Wet Waste Detected -> Wet Bin")
                    set_servo(S3, 140)
                    time.sleep(0.1)
                    set_servo(S0, 140)
                    time.sleep(3)
                    set_servo(S0, 0)

                else:
                    print("➡️ Dry Waste (Default)")
                    # Assuming Dry is S3=70 (Center) based on previous context, 
                    # OR user might want S3=0 (default in func) if not specified? 
                    # Reusing user's logic: "if noting is detected... move s0-140..."
                    # Since user code didn't set S3 in else block, it stays at last pos?
                    # The prompt says "if noting is detected no metal and no wet then move s0- 140 delay 3s then s0 - 0"
                    # It implies S3 stays or is irrelevant? But standard practice resets it.
                    # I will keep user's exact logic structure.
                    set_servo(S0, 140)
                    time.sleep(3)
                    set_servo(S0, 0)

                # Step 7 - Lift Up to Home
                print("⬆️ Lift UP to Home (L2)...")
                while GPIO.input(L2) == 1:
                    lift_up()
                    time.sleep(0.001)
                lift_stop()

                print("✅ Cycle Completed. Ready Again.")

            elif cmd.upper() == "Q":
                break

    except KeyboardInterrupt:
        print("\nForce Stopped.")
    finally:
        lift_stop()
        GPIO.cleanup()
