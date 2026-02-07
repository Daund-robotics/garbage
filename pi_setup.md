# Raspberry Pi 4B Setup Guide for Garbage Detection

This guide assumes you are starting with a fresh Raspberry Pi OS (64-bit recommended, Bookworm or later).

## 1. System Update & Dependencies

Open a terminal on your Raspberry Pi and run:

```bash
sudo apt update && sudo apt upgrade -y
# Install system libraries required by OpenCV
sudo apt install -y libgl1-mesa-glx libglib2.0-0 python3-venv git
```

## 2. Clone the Repository

```bash
git clone https://github.com/Daund-robotics/garbage.git
cd garbage
```

## 3. Automated Setup

We have provided a script to set up the virtual environment and install dependencies automatically.

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
1. Create a Python virtual environment (`venv`).
2. Install required Python packages (`opencv-python`, `cvzone`, `numpy`).
3. Ensure the environment is ready.

## 4. Run the Application

Once setup is complete, you can run the application:

```bash
chmod +x run.sh
./run.sh
```

### Notes
- **Model**: The project uses `yolov8n.onnx` (Nano model) which is optimized for speed on the Raspberry Pi CPU.
- **Performance**: Expect around 1-3 FPS on a standard Pi 4B CPU. For higher performance, an accelerator (like Hailo-8L) or further optimization (NCNN) would be needed, but ONNX is great for starting out.
- **Headless Mode**: If running without a monitor, `cv2.imshow` will fail. Ensure you have a display attached or use X11 forwarding.
- **USB Camera**: The script now automatically tries to connect to camera index 0 and then 1. If your camera is not detected:
  - Check connections.
  - Run `ls /dev/video*` to see if the device is recognized.
  - Ensure no other process is using the camera.
