import RPi.GPIO as GPIO
import time

# Pin Definitions
L1 = 5
L2 = 6

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(L1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(L2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Reading Limit Switches (Ctrl+C to stop)")
print(f"L1 Pin: {L1} | L2 Pin: {L2}")
print("Expected: 1 (HIGH) when NOT pressed, 0 (LOW) when PRESSED")
print("-" * 40)

try:
    while True:
        l1_val = GPIO.input(L1)
        l2_val = GPIO.input(L2)
        
        status_l1 = "PRESSED" if l1_val == 0 else "OPEN"
        status_l2 = "PRESSED" if l2_val == 0 else "OPEN"
        
        print(f"L1: {l1_val} ({status_l1}) | L2: {l2_val} ({status_l2})", end="\r")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting...")
    GPIO.cleanup()
