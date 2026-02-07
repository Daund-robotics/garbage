# Raspberry Pi 4B Setup Guide for Garbage Detection

This guide assumes you are starting with a fresh Raspberry Pi OS (64-bit recommended).

## 1. System Update & Dependencies

Open a terminal on your Raspberry Pi and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv libgl1-mesa-glx -y
```

## 2. Set Up Virtual Environment (Recommended)

It's best practice to use a virtual environment to avoid conflicts.

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Python Libraries

Install the required libraries. `ultralytics` handles the YOLO model, and `opencv-python-headless` is often better for Pi (lighter, no GUI dependencies, though we need GUI for `imshow` if you have a screen attached. If running headless, use `headless`).

```bash
pip install ultralytics
pip install opencv-python
pip install cvzone
pip install numpy
```

## 4. Export the Model for Speed (CRITICAL)

The standard `.pt` model is too slow for the Pi CPU. We will export it to **NCNN** format, which is optimized for ARM devices.

Run this command in your project folder on the Pi (or on your PC and transfer the resulting folder):

```bash
# This will create a directory named 'yolov8s_ncnn_model'
yolo export model=yolov8s.pt format=ncnn
```

> **Note**: If `yolov8s` is still too slow (< 1-2 FPS), try using the Nano model:
> `yolo export model=yolov8n.pt format=ncnn`

## 5. Run the Optimized Script

Transfer `main_pi.py` to your Pi and run it:

```bash
python main_pi.py
```
