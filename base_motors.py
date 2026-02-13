import RPi.GPIO as GPIO
import time

# ================= PIN DEFINITIONS =================
# Left Motor
IN1 = 27
IN2 = 17
ENA = 12

# Right Motor
IN3 = 22
IN4 = 23
ENB = 13

# ================= CONFIGURATION =================
SPEED = 255
TURN_SPEED = 200

pwm_a = None
pwm_b = None
initialized = False

def init():
    global pwm_a, pwm_b, initialized
    if initialized:
        return

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
    
    pwm_a = GPIO.PWM(ENA, 1000)
    pwm_b = GPIO.PWM(ENB, 1000)
    
    pwm_a.start(SPEED)
    pwm_b.start(SPEED)
    
    initialized = True
    print("✅ Base Motors Initialized")

def stop():
    if not initialized: init()
    GPIO.output([IN1, IN2, IN3, IN4], 0)

def forward():
    if not initialized: init()
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)
    GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    GPIO.output(IN3, 0); GPIO.output(IN4, 1)

def backward():
    if not initialized: init()
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)
    GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    GPIO.output(IN3, 1); GPIO.output(IN4, 0)

def left():
    if not initialized: init()
    pwm_a.ChangeDutyCycle(TURN_SPEED)
    pwm_b.ChangeDutyCycle(TURN_SPEED)
    GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    GPIO.output(IN3, 1); GPIO.output(IN4, 0)

def right():
    if not initialized: init()
    pwm_a.ChangeDutyCycle(TURN_SPEED)
    pwm_b.ChangeDutyCycle(TURN_SPEED)
    GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    GPIO.output(IN3, 0); GPIO.output(IN4, 1)

def cleanup():
    stop()
    if pwm_a: pwm_a.stop()
    if pwm_b: pwm_b.stop()
    GPIO.cleanup([IN1, IN2, IN3, IN4, ENA, ENB])
