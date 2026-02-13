# ============================================
# automation_pre_test.py  (INDUSTRIAL STABLE)
# ============================================

try:
    import smbus
except ImportError:
    print("Warning: smbus not found. Using mock for testing.")
    class MockSMBus:
        def __init__(self, bus): pass
        def write_byte_data(self, addr, reg, val): pass
        def read_byte_data(self, addr, reg): return 0
    smbus = type('obj', (object,), {'SMBus': MockSMBus})

import time

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
        def input(self, pin): return 0 # Simulate button press or not? 1 is unpressed usually
        def cleanup(self): pass
    GPIO = MockGPIO()

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ---- LIMIT SWITCHES ----
L1 = 5
L2 = 6

# ---- SENSOR PINS ----
WET_SENSOR_PIN   = 20
METAL_SENSOR_PIN = 16

# ---- MOTOR PINS ----
LIFT_UP   = 15
LIFT_DOWN = 14

GPIO.setup(L1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(L2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(WET_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(METAL_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LIFT_UP, GPIO.OUT)
GPIO.setup(LIFT_DOWN, GPIO.OUT)

# ================= PCA9685 =================
MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06

bus = smbus.SMBus(1)
PCA_ADDR = 0x40

SERVO_MIN = 102
SERVO_MAX = 512

DEFAULTS = {
    0: 30,
    1: 30,
    3: 90
}

# ================= PCA FUNCTIONS =================
def write_byte(reg, val):
    try:
        bus.write_byte_data(PCA_ADDR, reg, val)
    except Exception as e:
        print(f"Error writing to PCA9685: {e}")

def read_byte(reg):
    try:
        return bus.read_byte_data(PCA_ADDR, reg)
    except Exception as e:
        print(f"Error reading from PCA9685: {e}")
        return 0

def init_pca():
    try:
        write_byte(MODE1, 0x00)
        write_byte(MODE2, 0x04)
        time.sleep(0.01)
    except Exception as e:
        print(f"Error initializing PCA9685: {e}")

def set_pwm_freq(freq):
    try:
        prescaleval = 25000000.0 / 4096.0 / freq - 1
        prescale = int(prescaleval + 0.5)

        oldmode = read_byte(MODE1)
        write_byte(MODE1, (oldmode & 0x7F) | 0x10)
        write_byte(PRESCALE, prescale)
        write_byte(MODE1, oldmode)
        time.sleep(0.005)
        write_byte(MODE1, oldmode | 0x80)
    except Exception as e:
        print(f"Error setting PWM frequency: {e}")

def set_pwm(ch, on, off):
    try:
        base = LED0_ON_L + 4 * ch
        write_byte(base, on & 0xFF)
        write_byte(base+1, on >> 8)
        write_byte(base+2, off & 0xFF)
        write_byte(base+3, off >> 8)
    except Exception as e:
        print(f"Error setting PWM: {e}")

def angle_to_pwm(angle):
    pulse = SERVO_MIN + (angle/180.0)*(SERVO_MAX-SERVO_MIN)
    return int(pulse)

def move_servo(ch, angle):
    print(f"Servo {ch} -> {angle} deg")
    set_pwm(ch, 0, angle_to_pwm(angle))
    time.sleep(0.3)

# ================= MOTOR =================
def motor_stop():
    GPIO.output(LIFT_UP, 0)
    GPIO.output(LIFT_DOWN, 0)

def motor_up():
    GPIO.output(LIFT_UP, 1)
    GPIO.output(LIFT_DOWN, 0)

def motor_down():
    GPIO.output(LIFT_UP, 0)
    GPIO.output(LIFT_DOWN, 1)

# ================= MOVEMENT =================
def move_up_until_L2():
    motor_up()
    # No timeout - wait forever until L2 is pressed (LOW)
    while GPIO.input(L2) == 1:
        time.sleep(0.05)
    motor_stop()
    print("TOP Reached")

def move_down_until_L1(check_sensors=False):

    motor_down()

    metal_confirmed = False
    wet_confirmed = False

    metal_start = None
    wet_start = None

    REQUIRED_TIME = 3  # seconds stable detection required
    
    # Updated to handle potential NC switch or inverted logic
    # User reported L1 reads 'Pressed' (0) when not pressed. 
    # So we loop while it is 0, and stop when it becomes 1 (High).
    while GPIO.input(L1) == 0:

        if check_sensors:

            wet_raw   = GPIO.input(WET_SENSOR_PIN)
            metal_raw = GPIO.input(METAL_SENSOR_PIN)

            # ---- METAL LOGIC ----
            if wet_raw == 1 and metal_raw == 0:
                if metal_start is None:
                    metal_start = time.time()
                elif time.time() - metal_start >= REQUIRED_TIME:
                    metal_confirmed = True
            else:
                metal_start = None

            # ---- WET LOGIC ----
            if wet_raw == 0 and metal_raw == 1:
                if wet_start is None:
                    wet_start = time.time()
                elif time.time() - wet_start >= REQUIRED_TIME:
                    wet_confirmed = True
            else:
                wet_start = None

        time.sleep(0.05)

    motor_stop()
    print("BOTTOM Reached")

    return metal_confirmed, wet_confirmed

# ================= DEFAULT =================
def set_defaults():
    for ch, ang in DEFAULTS.items():
        move_servo(ch, ang)

# ================= AUTOMATION =================
def automation_sequence():

    print("\n===== AUTOMATION START =====")

    move_servo(1, 150)
    move_down_until_L1()
    move_servo(1, 30)

    move_up_until_L2()
    time.sleep(4)

    move_servo(1, 150)

    metal, wet = move_down_until_L1(check_sensors=True)

    print("Detection Result -> Metal:", metal, "| Wet:", wet)

    if metal:
        print("METAL CONFIRMED (3s stable)")
        move_servo(3, 20)
        move_servo(0, 140)
        time.sleep(3)
        move_servo(0, 30)
        move_servo(3, 90)

    elif wet:
        print("WET CONFIRMED (3s stable)")
        move_servo(3, 160)
        move_servo(0, 140)
        time.sleep(3)
        move_servo(0, 30)
        move_servo(3, 90)

    else:
        print("NO OBJECT CONFIRMED")
        move_servo(3, 90)
        move_servo(0, 140)
        time.sleep(3)
        move_servo(0, 30)
        move_servo(3, 90)

    move_up_until_L2()

    print("===== AUTOMATION COMPLETE =====\n")

# ================= MAIN =================
if __name__ == "__main__":

    print("SYSTEM START")

    init_pca()
    set_pwm_freq(50)
    time.sleep(0.5)

    set_defaults()

    if GPIO.input(L2) == 1:
        move_up_until_L2()
        set_defaults()

    print("Press N to start")

    try:
        while True:
            cmd = input(">> ")

            if cmd.lower() == "exit":
                break

            if cmd.upper() == "N":
                automation_sequence()

    except KeyboardInterrupt:
        pass

    motor_stop()
    GPIO.cleanup()
    print("Shutdown")
