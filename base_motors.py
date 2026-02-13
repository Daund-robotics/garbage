try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Warning: RPi.GPIO not found. Using mock for testing.")
    class MockGPIO:
        BCM = 10
        OUT = 0
        IN = 1
        PUD_UP = 2
        def setmode(self, mode): pass
        def setwarnings(self, flag): pass
        def setup(self, pin, mode, pull_up_down=None): pass
        def output(self, pin, state): pass
        def input(self, pin): return 0
        def cleanup(self, pins=None): pass
        class PWM:
            def __init__(self, pin, freq): pass
            def start(self, dc): pass
            def stop(self): pass
            def ChangeDutyCycle(self, dc): pass
    GPIO = MockGPIO()
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
SPEED = 100
TURN_SPEED = 90

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
    
    if pwm_a is None:
        pwm_a = GPIO.PWM(ENA, 1000)
        pwm_a.start(SPEED)
    
    if pwm_b is None:
        pwm_b = GPIO.PWM(ENB, 1000)
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
