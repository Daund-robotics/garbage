import os
import subprocess
import time
import signal
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# --- CONFIGURATION ---
TARGET_SCRIPT = "/home/rpi/garbage_4/garbage/main_pi_ultra.py"
# For testing on Windows/non-Pi, you can uncomment the below or set up a dummy
# TARGET_SCRIPT = "dummy_script.py" 

process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    global process
    is_running = False
    if process:
        if process.poll() is None:
            is_running = True
        else:
            # Process finished on its own
            process = None
    
    return jsonify({"running": is_running})

@app.route('/start', methods=['POST'])
def start_script():
    global process
    if process and process.poll() is None:
        return jsonify({"success": False, "message": "Already running"}), 400
    
    try:
        if not os.path.exists(TARGET_SCRIPT):
             return jsonify({"success": False, "message": f"File not found: {TARGET_SCRIPT}"}), 404

        # Run with python3 -u (unbuffered) to capture output if needed, 
        # but here we just fire and forget mostly.
        # process = subprocess.Popen(["python3", "-u", TARGET_SCRIPT], cwd=os.path.dirname(TARGET_SCRIPT))
        
        # Using cwd is safer for relative paths inside the script
        script_dir = os.path.dirname(TARGET_SCRIPT)
        process = subprocess.Popen(["python3", TARGET_SCRIPT], cwd=script_dir)
        
        return jsonify({"success": True, "message": "Started"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_script():
    global process
    if not process:
        return jsonify({"success": False, "message": "Not running"}), 400
    
    try:
        # SIGINT (CTRL+C) allows the script to cleanup if handlers are present
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill() # Force kill if it doesn't stop
            
        process = None
        return jsonify({"success": True, "message": "Stopped"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    # Host on 0.0.0.0 to be accessible via Hotspot
    app.run(host='0.0.0.0', port=5000)
