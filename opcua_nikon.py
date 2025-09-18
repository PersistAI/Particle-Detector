from opcua import Server, ua
import subprocess
import os
import time
import sys

# ==============================
# USER SETTINGS
# ==============================
DIGICAM_CMD = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
WAIT_SECONDS = 10  # ⏱️ Time to wait after trigger before running next script
NEXT_SCRIPT = "vialprogram1.py"  # Python script to run after capture
# ==============================

# 📂 Save directory (relative to this script folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "image_process_input")

# Ensure directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def trigger_handler(parent):
    """OPC UA callable: trigger DigiCamControl capture, wait, then run another script."""
    try:
        # Fire camera (non-blocking)
        subprocess.Popen(
            [DIGICAM_CMD, "/capture", "/dir", SAVE_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("📸 Capture triggered...")

        # Wait for images to download
        print(f"⏳ Waiting {WAIT_SECONDS} seconds...")
        time.sleep(WAIT_SECONDS)

        # Run next script
        script_path = os.path.join(BASE_DIR, NEXT_SCRIPT)
        if os.path.exists(script_path):
            print(f"▶️ Running next script: {NEXT_SCRIPT}")
            subprocess.Popen([sys.executable, script_path])
            return f"Capture done → waited {WAIT_SECONDS}s → ran {NEXT_SCRIPT}"
        else:
            return f"Capture done → waited {WAIT_SECONDS}s → script not found: {NEXT_SCRIPT}"

    except Exception as e:
        return f"Error: {e}"

# --- OPC UA Server Setup ---
server = Server()
server.set_endpoint("opc.tcp://0.0.0.0:4840/nikon_server/")

uri = "http://example.com/nikon"
idx = server.register_namespace(uri)

# Camera object
camera = server.nodes.objects.add_object(idx, "NikonD800E")

# Add TriggerCapture method
camera.add_method(
    idx,
    "TriggerCapture",
    trigger_handler,
    [], [ua.VariantType.String]
)

# --- Run Server ---
if __name__ == "__main__":
    server.start()
    print("✅ OPC UA Nikon server running at opc.tcp://0.0.0.0:4840/nikon_server/")
    try:
        while True:
            time.sleep(1)
    finally:
        server.stop()
