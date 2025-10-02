# opcua_camera_server.py
from opcua import Server, ua
from mecademicpy.robot import Robot
import mecca_moves_complete           # Particle imaging
import mecca_moves_complete2          # Phase separation
import subprocess
import os
import time
import threading
import sys

# ==============================
# USER SETTINGS
# ==============================
DIGICAM_CMD     = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
ROBOT_IP        = "192.168.0.100"

SEQUENCE_FILE   = "sequences_dualmode.txt"


# Script to trigger after each photo
POST_PHOTO_SCRIPT = "vialprogram1.py"      # for particles
POST_PHOTO_SCRIPT_PHASE = "vialprogram2.py" # for phase separation
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "image_process_input")
SEQ_PATH = os.path.join(BASE_DIR, SEQUENCE_FILE)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Camera trigger ---
def fire_camera():
    try:
        print("📸 [Camera] Firing DigiCamControl...")
        subprocess.Popen(
            [DIGICAM_CMD, "/capture", "/dir", SAVE_DIR, "/wait", "5000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        print("✅ [Camera] Trigger sent.")
    except Exception as e:
        print(f"❌ [Camera] Trigger error: {e}")

# --- OPC UA setup ---
server = Server()
server.set_endpoint("opc.tcp://0.0.0.0:4840/nikon_server/")

uri = "http://example.com/nikon"
idx = server.register_namespace(uri)
root = server.nodes.objects
cell = root.add_object(idx, "PhotoCell")

# Persistent robot connection
robot = Robot()

def _robot_connect_once():
    print(f"🔗 Connecting to Mecademic at {ROBOT_IP} ...")
    robot.Connect(ROBOT_IP)
    robot.WaitConnected()
    robot.ActivateRobot()
    robot.Home()
    robot.WaitHomed()
    print("✅ Robot connected, homed, and held (persistent).")

# --- Background runner infra ---
_run_lock = threading.Lock()
_is_running = False

def _launch_job(fn):
    """Launch a background run with exclusive lock."""
    global _is_running
    with _run_lock:
        if _is_running:
            print("⚠️ A run is already in progress; ignoring new trigger.")
            return
        _is_running = True

    def _job():
        global _is_running
        try:
            fn()
        except Exception as e:
            print(f"❌ Background job error: {e}")
        finally:
            with _run_lock:
                _is_running = False
            print("ℹ️ Ready for next command.")

    threading.Thread(target=_job, daemon=True).start()

# --- Methods exposed over OPC UA ---

def ua_RunAll(parent, limit):
    """Run using mecca_moves_complete (particle analysis)."""
    def _do():
        sequences = mecca_moves_complete.load_sequences(SEQ_PATH)
        if not sequences:
            print("❌ No sequences loaded; aborting RunAll.")
            return

        try:
            max_pos = int(limit.Value if hasattr(limit, "Value") else limit)
        except Exception as e:
            print(f"⚠️ Could not parse limit: {e}")
            max_pos = None

        # 🔥 Now just hand off — mecca_moves_complete handles its own timing vars
        mecca_moves_complete.run_sequences(
            robot=robot,
            sequences=sequences,
            camera_trigger=fire_camera,
            post_photo_script=POST_PHOTO_SCRIPT,
            max_positions=max_pos
        )
        print("✅ RunAll complete (particle analysis).")
    _launch_job(_do)
    return None


def ua_RunAllPhase(parent, limit):
    """Run using mecca_moves_complete2 (phase separation analysis)."""
    def _do():
        sequences = mecca_moves_complete2.load_sequences(SEQ_PATH)
        if not sequences:
            print("❌ No sequences loaded; aborting RunAllPhase.")
            return

        try:
            max_pos = int(limit.Value if hasattr(limit, "Value") else limit)
        except Exception as e:
            print(f"⚠️ Could not parse limit: {e}")
            max_pos = None

        # 🔥 Just hand off — mecca_moves_complete2 owns its own waits/vars
        mecca_moves_complete2.run_sequences(
            robot=robot,
            sequences=sequences,
            camera_trigger=fire_camera,
            post_photo_script=POST_PHOTO_SCRIPT_PHASE,
            max_positions=max_pos
        )
        print("✅ RunAllPhase complete (phase separation).")
    _launch_job(_do)
    return None


# Register all methods
cell.add_method(idx, "RunAll", ua_RunAll, [ua.VariantType.Int32], [])
cell.add_method(idx, "RunAllPhase", ua_RunAllPhase, [ua.VariantType.Int32], [])

if __name__ == "__main__":
    _robot_connect_once()
    server.start()
    print("✅ OPC UA server at opc.tcp://0.0.0.0:4840/nikon_server/")
    try:
        while True:
            time.sleep(1)
    finally:
        server.stop()
