import time
import RPi.GPIO as GPIO
import automation_pre_test
import base_motors

# --- PIN CONFIGURATION ---
GPIO_TRIGGER = 8
GPIO_ECHO = 7

# --- SETTINGS ---
TARGET_MIN = 10   # cm
TARGET_MAX = 15   # cm
FAR_LIMIT = 70    # cm
STABILITY_TIME = 2.0 # seconds to wait before moving forward when object is far

# --- SETUP GPIO ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def get_distance():
    # Ensure trigger is low
    GPIO.output(GPIO_TRIGGER, False)
    time.sleep(0.00001) # Short delay

    # Send 10us pulse
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    start_time = time.time()
    stop_time = time.time()
    
    # Timeout for safety (avoid infinite loop)
    timeout = time.time() + 0.1

    # Wait for Echo HIGH
    while GPIO.input(GPIO_ECHO) == 0:
        start_time = time.time()
        if time.time() > timeout:
            return -1

    # Wait for Echo LOW
    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = time.time()
        if time.time() > timeout:
            return -1

    elapsed = stop_time - start_time
    distance = (elapsed * 34300) / 2
    return distance

def main():
    print("Starting Ultrasonic Automation System...")
    
    # 1. INITIALIZE ALL SYSTEMS
    try:
        print("Initializing Automation & Motors...")
        base_motors.init()        # Ensure motors are ready (Speed is 100 in base_motors.py?)
        automation_pre_test.init_pca()
        automation_pre_test.set_pwm_freq(50)
        time.sleep(0.5)
        
        # Set Defaults
        automation_pre_test.set_defaults()
        print("Set Defaults Done.")
        
        # Move to Top
        print("Moving Lift to TOP position...")
        automation_pre_test.move_up_until_L2()
        
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    print("Initialization Complete. Entering Control Loop.")
    print(f"Target Range: {TARGET_MIN}-{TARGET_MAX} cm")

    far_start_time = None
    
    try:
        while True:
            dist = get_distance()
            
            if dist == -1:
                # print("Sensor Timeout")
                continue

            print(f"Distance: {dist:.1f} cm", end="\r")

            # --- LOGIC ---
            
            # CASE 1: TOO CLOSE (< 10cm)
            if dist < TARGET_MIN:
                print(f"\nToo Close ({dist:.1f}cm)! Moving BACKWARD.")
                base_motors.backward()
                far_start_time = None # Reset stability timer

            # CASE 2: IN TARGET RANGE (10cm - 15cm)
            elif TARGET_MIN <= dist <= TARGET_MAX:
                print(f"\nIn Range ({dist:.1f}cm)! Starting AUTOMATION.")
                base_motors.stop()
                
                # Execute Automation
                automation_pre_test.automation_sequence()
                
                print("Automation Finished. Resuming monitoring...")
                # Optional: Add delay or move back after automation to avoid immediate re-trigger?
                time.sleep(2) 
                # Re-init defaults just in case
                automation_pre_test.set_defaults()
                automation_pre_test.move_up_until_L2()
                
                far_start_time = None

            # CASE 3: FAR BUT DETECTED (15cm - 70cm)
            elif TARGET_MAX < dist < FAR_LIMIT:
                # Check for stability
                if far_start_time is None:
                    far_start_time = time.time()
                    base_motors.stop() # Wait to confirm
                
                elapsed = time.time() - far_start_time
                
                if elapsed >= STABILITY_TIME:
                    print(f"\nTarget Sighted ({dist:.1f}cm). Moving FORWARD.")
                    base_motors.forward()
                else:
                    print(f"Waiting for stability... {elapsed:.1f}s", end="\r")
                    base_motors.stop()

            # CASE 4: OUT OF RANGE (> 70cm)
            else:
                base_motors.stop()
                far_start_time = None
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by User.")
        base_motors.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
