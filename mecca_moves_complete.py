# mecca_moves_complete.py
import time
import os
import subprocess
import sys

# ==============================
# Movement & photo timing knobs
# ==============================
MOVE_WAIT       = 2.0
RUN_VECTOR      = [ 0, 2, 3, 4, 5, 6, 4, 3, 1, 0]

PHOTO_PREP_SEQ  = [5]
PHOTO_PREP_WAIT = 2.45
PHOTO_SEQ       = [6]
PHOTO_WAIT      = 1.20

# Grid layout
ROWS            = 4       # A..D
COLS            = 6       # 1..6
X_SPACING       = 20.25    # A→D (rows, X+ direction)
Y_SPACING       = 20.20    # 1→6 (cols, Y+ direction)
# ==============================

# --- Sequence Loader (same as mecca_moves) ---
def _parse_value(tok):
    try:
        return float(tok.split("=")[1])
    except Exception:
        return None

def load_sequences(seq_path):
    """Load base sequences from file, including type and gripper state."""
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

    print(f"✅ Loaded {len(sequences)} base sequences.")
    return sequences

# --- Offset Logic ---
def _apply_offset_to_point(wp, dx, dy):
    """Return a new waypoint with X,Y offsets applied if Cartesian."""
    if wp["type"] == "cartesian":
        new_data = wp["data"].copy()
        new_data[0] += dx   # X
        new_data[1] += dy   # Y
        return {"type": "cartesian", "data": new_data, "grip": wp["grip"]}
    else:
        return wp  # leave joint moves unchanged

def _offset_sequences(base_sequences, dx, dy):
    """Apply XY offsets to sequences 0,1,2,3 only."""
    seqs = {}
    for key, seq in base_sequences.items():
        if key in (0, 1, 2, 3):
            pts = [_apply_offset_to_point(wp, dx, dy) for wp in seq["points"]]
            seqs[key] = {"name": seq["name"] + f" (offset {dx},{dy})", "points": pts}
        else:
            seqs[key] = seq
    return seqs

def generate_grid_sequences(base_sequences):
    """
    Expand base sequences into a ROWS×COLS grid (e.g. A1..D6).
    Columns = Y+, Rows = X+.
    """
    grid_runs = []
    for row in range(ROWS):     # A..D
        for col in range(COLS): # 1..6
            dx = row * X_SPACING
            dy = col * Y_SPACING
            label = f"{chr(ord('A')+row)}{col+1}"
            seqs = _offset_sequences(base_sequences, dx, dy)
            grid_runs.append((label, seqs))
    return grid_runs

# --- Runner ---
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
    Run through all vial positions defined by ROWS×COLS grid.
    For each position, run the full run_vector sequence list with offsets.
    """
    grid = generate_grid_sequences(sequences)

    for label, seqs in grid:
        print(f"\n=== ▶ Starting vial position {label} ===")
        order = run_vector if run_vector is not None else sorted(seqs.keys())
        photo_prep = set(photo_prep_seq if isinstance(photo_prep_seq, list) else [photo_prep_seq])
        photo_pose = set(photo_seq if isinstance(photo_seq, list) else [photo_seq])

        for key in order:
            if key not in seqs:
                print(f"⚠️ Sequence {key} missing, skipping")
                continue
            seq = seqs[key]
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

                # Launch extra script
                if post_photo_script:
                    try:
                        script_path = os.path.join(os.path.dirname(__file__), post_photo_script)
                        print(f"▶️ Launching {post_photo_script} ...")
                        subprocess.Popen([sys.executable, script_path])
                    except Exception as e:
                        print(f"⚠️ Failed to launch {post_photo_script}: {e}")

        print(f"=== ✅ Finished vial position {label} ===")

    print("\n✅ All vial positions complete (robot remains connected).")
