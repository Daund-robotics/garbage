import RPi.GPIO as GPIO
import time

# GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin Definitions
# User specified "wet" sensor, which is Pin 20 in existing code.
# We assume Pin 20 is ECHO. 
# We default TRIG to 21 (please change if connected elsewhere).
GPIO_TRIGGER = 21
GPIO_ECHO = 20

print(f"Ultrasonic Measurement - TRIG: {GPIO_TRIGGER}, ECHO: {GPIO_ECHO}")

# Set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance
 
if __name__ == '__main__':
    try:
        while True:
            dist = distance()
            print(f"Measured Distance = {dist:.1f} cm")
            time.sleep(1)
 
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
