# opcua_camera_server.py
from opcua import Server
from mecademicpy.robot import Robot
import mecca_moves
import mecca_moves_complete   # <--- new module for RunAll
import subprocess
import os
import time
import threading

# ==============================
# USER SETTINGS
# ==============================
DIGICAM_CMD     = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
ROBOT_IP        = "192.168.0.100"

SEQUENCE_FILE   = "sequences_dualmode.txt"

# Script to trigger after each photo
POST_PHOTO_SCRIPT = None
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

def ua_Run(parent):
    """Run using mecca_moves (single-shot tuned version)."""
    def _do():
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
            post_photo_script=POST_PHOTO_SCRIPT,
        )
        print("✅ Run complete.")
    _launch_job(_do)
    return None


def ua_RunAll(parent):
    """Run using mecca_moves_complete (e.g. full grid)."""
    def _do():
        sequences = mecca_moves_complete.load_sequences(SEQ_PATH)
        if not sequences:
            print("❌ No sequences loaded; aborting RunAll.")
            return

        mecca_moves_complete.run_sequences(
            robot=robot,
            sequences=sequences,
            run_vector=mecca_moves_complete.RUN_VECTOR,
            move_wait=mecca_moves_complete.MOVE_WAIT,
            photo_prep_seq=mecca_moves_complete.PHOTO_PREP_SEQ,
            photo_prep_wait=mecca_moves_complete.PHOTO_PREP_WAIT,
            photo_seq=mecca_moves_complete.PHOTO_SEQ,
            photo_wait=mecca_moves_complete.PHOTO_WAIT,
            camera_trigger=fire_camera,
            post_photo_script=POST_PHOTO_SCRIPT,
        )
        print("✅ RunAll complete.")
    _launch_job(_do)
    return None
# Register both methods
cell.add_method(idx, "Run", ua_Run, [], [])
cell.add_method(idx, "RunAll", ua_RunAll, [], [])

if __name__ == "__main__":
    _robot_connect_once()
    server.start()
    print("✅ OPC UA server at opc.tcp://0.0.0.0:4840/nikon_server/")
    try:
        while True:
            time.sleep(1)
    finally:
        server.stop()