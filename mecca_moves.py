# mecca_moves.py
import time
import os
import subprocess
import sys

def _parse_value(tok):
    try:
        return float(tok.split("=")[1])
    except Exception:
        return None

def load_sequences(seq_path):
    """Load sequences from file, including type and gripper state."""
    sequences = {}
    if not os.path.exists(seq_path):
        print(f"❌ Sequence file not found: {seq_path}")
        return sequences

    with open(seq_path, "r", encoding="utf-8") as f:
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

    print(f"✅ Loaded {len(sequences)} sequences.")
    return sequences


def run_sequences(robot,
                  sequences: dict,
                  run_vector,
                  move_wait: float,
                  photo_prep_seq,
                  photo_prep_wait,
                  photo_seq,
                  photo_wait,
                  camera_trigger,
                  post_photo_script=None):
    """
    Runs sequences with simple camera timing:
      - Execute waypoints (with move_wait per waypoint)
      - Trigger camera at PHOTO_PREP_SEQ, wait PHOTO_PREP_WAIT
      - Hold pose at PHOTO_SEQ, wait PHOTO_WAIT
      - Optionally launch post_photo_script after PHOTO_WAIT
    """
    order = run_vector if run_vector is not None else sorted(sequences.keys())
    photo_prep = set(photo_prep_seq if isinstance(photo_prep_seq, list) else [photo_prep_seq])
    photo_pose = set(photo_seq if isinstance(photo_seq, list) else [photo_seq])

    for key in order:
        if key not in sequences:
            print(f"⚠️ Sequence {key} missing, skipping")
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

            if move_wait and move_wait > 0:
                time.sleep(move_wait)

        # Fire camera at prep step
        if key in photo_prep:
            print(f"⚡ [PhotoPrep] Trigger camera at Sequence {key}")
            try:
                camera_trigger()
            except Exception as e:
                print(f"⚠️ camera_trigger() failed: {e}")
            if photo_prep_wait and photo_prep_wait > 0:
                print(f"⏸️ [PhotoPrep] Waiting {photo_prep_wait}s")
                time.sleep(photo_prep_wait)

        # Hold at photo pose
        if key in photo_pose:
            if photo_wait and photo_wait > 0:
                print(f"📸 [PhotoPose] Holding at Sequence {key} for {photo_wait}s")
                time.sleep(photo_wait)

            # Launch extra script once photo hold is done
            if post_photo_script:
                try:
                    script_path = os.path.join(os.path.dirname(__file__), post_photo_script)
                    print(f"▶️ Launching {post_photo_script} ...")
                    subprocess.Popen([sys.executable, script_path])
                except Exception as e:
                    print(f"⚠️ Failed to launch {post_photo_script}: {e}")

    print("\n✅ All sequences complete (robot remains connected).")

