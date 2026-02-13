import RPi.GPIO as GPIO
import time

# Pin Definitions
L1 = 5
L2 = 6

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# Using PUD_UP: Input is HIGH (1) when open, LOW (0) when pressed (connected to GND)
GPIO.setup(L1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(L2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Starting Limit Switch Test (limit_test.py)")
print(f"Reading Pins: L1={L1}, L2={L2}")
print("Press Ctrl+C to exit.")
print("-" * 30)

try:
    while True:
        l1_val = GPIO.input(L1)
        l2_val = GPIO.input(L2)
        
        # Interpret values
        # Assumes switch connects Pin to GND when pressed
        state_l1 = "PRESSED" if l1_val == 0 else "OPEN"
        state_l2 = "PRESSED" if l2_val == 0 else "OPEN"
        
        # Print with carriage return to update same line
        print(f"L1: {l1_val} [{state_l1}]  |  L2: {l2_val} [{state_l2}]   ", end="\r")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTest Stopped.")
    GPIO.cleanup()
