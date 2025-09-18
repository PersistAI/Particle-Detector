# opcua_camera_server.py
from opcua import Server
from mecademicpy.robot import Robot
import mecca_moves
import subprocess
import os
import time
import threading

# ==============================
# USER SETTINGS
# ==============================
DIGICAM_CMD     = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
ROBOT_IP        = "192.168.0.100"

SEQUENCE_FILE   = "sequences_dualmode.txt"  # parsed inside mecca_moves.load_sequences

# Movement & photo timing knobs
MOVE_WAIT       = 2.0     # per-waypoint wait (seconds)
RUN_VECTOR      = [0, 1, 2, 3, 4, 5, 6, 4, 3, 1, 0]

PHOTO_PREP_SEQ  = [5]     # where we TRIGGER DigiCam
PHOTO_PREP_WAIT = 2.45    # wait after trigger before moving on
PHOTO_SEQ       = [6]     # where robot HOLDS pose
PHOTO_WAIT      = 2.0    # hold time at photo pose
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "image_process_input")
SEQ_PATH = os.path.join(BASE_DIR, SEQUENCE_FILE)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Camera trigger (fire-and-forget) ---
def fire_camera():
    """Tell DigiCamControl to capture (non-blocking)."""
    try:
        print("📸 [Camera] Firing DigiCamControl...")
        subprocess.Popen(
            [DIGICAM_CMD, "/capture", "/dir", SAVE_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)  # quiet on Windows
        )
        print("✅ [Camera] Trigger sent.")
    except Exception as e:
        print(f"❌ [Camera] Trigger error: {e}")

# --- OPC UA server setup ---
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

# --- Background runner so OPC UA method returns immediately ---
_run_lock = threading.Lock()
_is_running = False

def _run_job():
    global _is_running
    try:
        sequences = mecca_moves.load_sequences(SEQ_PATH)
        if not sequences:
            print("❌ No sequences loaded; aborting run.")
            return

        mecca_moves.run_sequences(
            robot=robot,
            sequences=sequences,
            run_vector=RUN_VECTOR,
            move_wait=MOVE_WAIT,
            photo_prep_seq=PHOTO_PREP_SEQ,
            photo_prep_wait=PHOTO_PREP_WAIT,
            photo_seq=PHOTO_SEQ,
            photo_wait=PHOTO_WAIT,
            camera_trigger=fire_camera,
        )
        print("✅ Run complete. Robot remains connected and stiff.")
    except Exception as e:
        print(f"❌ Background run error: {e}")
    finally:
        with _run_lock:
            global _is_running
            _is_running = False
        print("ℹ️ Ready for next Run.")

def ua_Run(parent):
    """Fire-and-forget: start a run in the background and return immediately."""
    global _is_running
    with _run_lock:
        if _is_running:
            print("⚠️ A run is already in progress; ignoring new trigger.")
            return None
        _is_running = True
    threading.Thread(target=_run_job, daemon=True).start()
    # Return immediately so OPC UA client never times out
    return None

# Register method (no strong typing → empty arg lists)
cell.add_method(idx, "Run", ua_Run, [], [])

if __name__ == "__main__":
    _robot_connect_once()
    server.start()
    print("✅ OPC UA server at opc.tcp://0.0.0.0:4840/nikon_server/")
    try:
        while True:
            time.sleep(1)
    finally:
        # Optional cleanup on shutdown (leave commented if you want to stay stiff)
        # try:
        #     robot.DeactivateRobot()
        #     robot.Disconnect()
        # except Exception as e:
        #     print(f"⚠️ Shutdown cleanup issue: {e}")
        server.stop()
