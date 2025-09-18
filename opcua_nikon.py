from opcua import Server
import subprocess
import os
import time
import sys
import socket
import threading

# ==============================
# USER SETTINGS
# ==============================
DIGICAM_CMD = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
MECCA_PORT = 50007    # TCP port for Mecca script to connect
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "image_process_input")
os.makedirs(SAVE_DIR, exist_ok=True)


# --- socket server for mecca communication ---
def mecca_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", MECCA_PORT))
        s.listen(1)
        print(f"📡 Mecca listener active on port {MECCA_PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                msg = conn.recv(1024).decode("utf-8").strip()
                print(f"🤖 From Mecca: {msg}")
                if msg == "READY_FOR_PHOTO":
                    do_capture()   # blocks until DigiCamControl is done
                    conn.sendall(b"PHOTO_DONE")


def do_capture():
    """Trigger DigiCamControl capture and wait until finished."""
    try:
        print("📸 Capture triggered, waiting for DigiCamControl...")
        result = subprocess.run(
            [DIGICAM_CMD, "/capture", "/dir", SAVE_DIR],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print("✅ Capture complete, image saved.")
        else:
            print(f"❌ Capture failed (code {result.returncode}): {result.stderr}")
    except Exception as e:
        print(f"❌ Camera error: {e}")


def trigger_handler(parent):
    """OPC UA callable: launch meccamovement.py and wait for sync."""
    try:
        print("▶️ Launching meccamovement.py ...")
        subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "meccamovement.py")])
        return None   # no string return → avoids VariantType error
    except Exception as e:
        print(f"Error launching meccamovement.py: {e}")
        return None


# --- OPC UA Server Setup ---
server = Server()
server.set_endpoint("opc.tcp://0.0.0.0:4840/nikon_server/")

uri = "http://example.com/nikon"
idx = server.register_namespace(uri)

camera = server.nodes.objects.add_object(idx, "NikonD800E")
camera.add_method(idx, "TriggerMeccaAndCapture", trigger_handler, [], [])

if __name__ == "__main__":
    threading.Thread(target=mecca_listener, daemon=True).start()
    server.start()
    print("✅ OPC UA Nikon server running at opc.tcp://0.0.0.0:4840/nikon_server/")
    try:
        while True:
            time.sleep(1)
    finally:
        server.stop()
