import time
import os
from mecademicpy.robot import Robot

# =====================
# CONFIG
ROBOT_IP     = "192.168.0.100"
SEQUENCE_FILE = "sequences_dualmode.txt"

MOVE_WAIT   = 2.0   # seconds to wait after each movement
RUN_VECTOR  = [0, 1, 2, 3, 4, 5, 6, 4, 3, 1, 0]  # order of sequences (set None for all in order)

PHOTO_SEQ   = [6]   # sequence(s) where photos are taken
PHOTO_WAIT  = 4.0      # wait (s) only after those sequences
# =====================

def _parse_value(tok):
    try:
        return float(tok.split("=")[1])
    except Exception:
        return None

def load_sequences():
    """Load sequences from file, including type and gripper state."""
    sequences = {}
    if not os.path.exists(SEQUENCE_FILE):
        print(f"❌ Sequence file not found: {SEQUENCE_FILE}")
        return sequences

    with open(SEQUENCE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_key, current_name, current_points = None, None, []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("==== SEQUENCE"):
            if current_key is not None:
                sequences[current_key] = {"name": current_name, "points": current_points}
                current_points = []
            parts = line.split()
            current_key = int(parts[2])
            current_name = f"Sequence {current_key}"
        else:
            segs = [seg.strip() for seg in line.split(",")]
            type_seg = next((s for s in segs if s.lower().startswith("type=")), None)
            grip_seg = next((s for s in segs if s.lower().startswith("gripper=")), None)
            if not type_seg or not grip_seg:
                continue
            wtype = type_seg.split("=")[1].strip().lower()
            grip  = grip_seg.split("=")[1].strip()

            if wtype == "cartesian":
                vals = {}
                for key in ["X","Y","Z","α","β","γ"]:
                    s = next((s for s in segs if s.startswith(f"{key}=")), None)
                    if s: vals[key] = _parse_value(s)
                if len(vals) == 6:
                    data = [vals["X"], vals["Y"], vals["Z"], vals["α"], vals["β"], vals["γ"]]
                    current_points.append({"type":"cartesian","data":data,"grip":grip})
            else:
                vals = {}
                for i in range(6):
                    key = f"J{i+1}"
                    s = next((s for s in segs if s.startswith(f"{key}=")), None)
                    if s: vals[key] = _parse_value(s)
                if len(vals) == 6:
                    data = [vals[f"J{i+1}"] for i in range(6)]
                    current_points.append({"type":"joints","data":data,"grip":grip})

    if current_key is not None:
        sequences[current_key] = {"name": current_name, "points": current_points}

    print(f"✅ Loaded {len(sequences)} sequences from {SEQUENCE_FILE}")
    return sequences


def run_sequences(robot, sequences):
    """Run through sequences in chosen order."""
    if RUN_VECTOR is not None:
        order = RUN_VECTOR
    else:
        order = sorted(sequences.keys())

    # normalize PHOTO_SEQ to list
    photo_targets = PHOTO_SEQ if isinstance(PHOTO_SEQ, list) else [PHOTO_SEQ]

    for key in order:
        if key not in sequences:
            print(f"⚠️ Sequence {key} not found, skipping")
            continue
        seq = sequences[key]
        print(f"\n▶ Running Sequence {key}: {seq['name']}")
        for i, wp in enumerate(seq["points"]):
            wtype, data, grip = wp["type"], wp["data"], wp["grip"]

            try:
                if wtype == "cartesian":
                    robot.MoveLin(*data)
                else:
                    robot.MoveJoints(*data)
            except Exception as e:
                print(f"⚠️ Move rejected at step {i+1}: {e}")

            if grip == "Open":
                robot.GripperOpen()
            elif grip == "Closed":
                robot.GripperClose()

            print(f"  Step {i+1}/{len(seq['points'])}: {wtype.upper()} → {data}, Gripper={grip}")
            time.sleep(MOVE_WAIT)

        # === Photo pause if designated ===
        if key in photo_targets:
            print(f"📸 Pausing {PHOTO_WAIT}s at Sequence {key} for photo...")
            time.sleep(PHOTO_WAIT)

    print("\n✅ All sequences complete.")


def main():
    robot = Robot()
    robot.Connect(ROBOT_IP)
    robot.WaitConnected()
    robot.ActivateRobot()
    robot.Home()
    robot.WaitHomed()

    sequences = load_sequences()
    if not sequences:
        return

    run_sequences(robot, sequences)

    # Keep robot active (don’t disconnect)
    print("\n🤖 Robot is still active and connected. Ready for next command.")


if __name__ == "__main__":
    main()
