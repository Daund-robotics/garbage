# Waste Expert Robot - Pin Connections

This document lists all the wiring connections based on the `automation_pre_test.py` script.

## 1. GPIO Sensors (Raspberry Pi Header)

| Component | Pin Function | GPIO Pin | Wiring Note |
| :--- | :--- | :--- | :--- |
| **Limit Switch 1 (Bottom)** | Input (Pull-Up) | **GPIO 5** | Connect between Pin & GND |
| **Limit Switch 2 (Top)** | Input (Pull-Up) | **GPIO 6** | Connect between Pin & GND |
| **Metal Sensor** | Input (Pull-Up) | **GPIO 16** | LOW when Metal Detected |
| **Wet Sensor** | Input (Pull-Up) | **GPIO 20** | LOW when Wet Detected |

## 2. Lift Motor (L298N Driver)

| Function | L298N Pin | GPIO Pin |
| :--- | :--- | :--- |
| **Lift UP** | INX (e.g., IN1) | **GPIO 15** |
| **Lift DOWN** | INY (e.g., IN2) | **GPIO 14** |

## 3. Servos (PCA9685 Driver)

**Connection:** I2C Bus (SDA/SCL) on Raspberry Pi.

| Servo Name | Channel | Function | Angles |
| :--- | :--- | :--- | :--- |
| **S0** | **Ch 0** | Garbage Type Dumper | 0° (Rest), 140° (Dump) |
| **S1** | **Ch 1** | Pick Bucket | 30° (Close/Grab), 150° (Open) |
| **S3** | **Ch 3** | Sorting Gate | 0° (Metal), 140° (Wet), 70° (Dry) |

## 4. Power Connections

*   **Raspberry Pi:** 5V USB-C
*   **PCA9685:** 5V (VCC) & Ground to Pi. **External 5V/6V Power** to Terminal Block for Servos.
*   **L298N:** External 12V/Battery Power for Motor. Ground shared with Pi.
