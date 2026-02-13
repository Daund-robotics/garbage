import cv2
import socket
import struct
import pickle
import threading
import time
import sys

# Import Hardware Modules
try:
    import base_motors
    import automation_pre_test
except ImportError:
    print("⚠️ Hardware modules not found. Running in mock mode.")
    import base_motors # uses mock from previous steps
    import automation_pre_test

# Configuration
VIDEO_PORT = 5555
CMD_PORT = 5556
BUFFER_SIZE = 4096

def init_hardware():
    print("🤖 Initializing Hardware...")
    try:
        # 1. Init Automation (Servos/PCA)
        automation_pre_test.init_pca()
        automation_pre_test.set_pwm_freq(50)
        time.sleep(0.5)
        automation_pre_test.set_defaults()
        
        # 2. Home Lift
        print("🔝 Homing Lift...")
        automation_pre_test.move_up_until_L2()
        
        # 3. Init Base Motors
        base_motors.init()
        print("✅ Hardware Ready.")
    except Exception as e:
        print(f"❌ Hardware Init Error: {e}")

def video_stream_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', VIDEO_PORT)) # Listen on all interfaces
    server_socket.listen(5)
    print(f"📷 Video Stream Server listening on port {VIDEO_PORT}")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        client_socket, addr = server_socket.accept()
        print(f"📷 Video Connected to: {addr}")
        
        if not cap.isOpened():
             cap.open(0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Compress frame
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                data = pickle.dumps(buffer)
                
                # Send message length first, then data
                message_size = struct.pack("Q", len(data))
                client_socket.sendall(message_size + data)
                
        except Exception as e:
            print(f"📷 Video Stream Error/Disconnect: {e}")
        finally:
            client_socket.close()

def command_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', CMD_PORT))
    server_socket.listen(5)
    print(f"🎮 Command Server listening on port {CMD_PORT}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"🎮 Command Connected to: {addr}")

        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                command = data.decode('utf-8').strip().upper()
                # print(f"Received Command: {command}") # Debug: print every command?
                
                if command == "FORWARD":
                    base_motors.forward()
                elif command == "BACKWARD":
                    base_motors.backward()
                elif command == "LEFT":
                    base_motors.left()
                elif command == "RIGHT":
                    base_motors.right()
                elif command == "STOP":
                    base_motors.stop()
                elif command == "AUTO":
                    print("🚀 Triggering Automation Sequence")
                    base_motors.stop() # Ensure stop before auto
                    # Run automation in a separate thread so we don't block command loop?
                    # Or block to prevent other commands? 
                    # Blocking is safer to avoid conflict.
                    try:
                        automation_pre_test.automation_sequence()
                    except Exception as e:
                        print(f"❌ Automation Error: {e}")
                    # client_socket.sendall(b"AUTO_DONE") # Optional: Ack
                else:
                    print(f"❓ Unknown Command: {command}")

        except Exception as e:
            print(f"🎮 Command Connection Error/Disconnect: {e}")
            base_motors.stop()
        finally:
            client_socket.close()
            base_motors.stop()

def main():
    init_hardware()

    # Start Video Thread
    t_video = threading.Thread(target=video_stream_server)
    t_video.daemon = True
    t_video.start()

    # Start Command Thread
    t_cmd = threading.Thread(target=command_server)
    t_cmd.daemon = True
    t_cmd.start()

    print("Running... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            # Find IP to display
            # A bit hacky to find actual IP
            pass
    except KeyboardInterrupt:
        print("Stopping...")
        base_motors.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main()
