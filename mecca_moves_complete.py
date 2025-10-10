import time
import os
import subprocess
import sys

# ==============================
# Movement & photo timing knobs
# ==============================
MOVE_WAIT       = 2.0
RUN_VECTOR      = [0, 2, 3, 4, 5, 6, 4, 3, 1, 0] 

PHOTO_PREP_SEQ  = [5]
PHOTO_PREP_WAIT = 2.00
PHOTO_SEQ       = [6]
PHOTO_WAIT      = 2.0

# Grid layout
ROWS            = 4       # A..D
COLS            = 6       # 1..6
X_SPACING       = 20.56   # A→D (rows, X+ direction)
Y_SPACING       = 20.4    # 1→6 (cols, Y+ direction)

# Safe sequence for run start/end
SAFE_SEQ = 4  # set to None to disable

# ==============================
# Custom offsets for specific positions
# Example: {"A2": (0.5, -0.3), "D6": (-1.0, 0.2)}
CUSTOM_OFFSETS = {"A2": (0.5, -0.4)}
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WATCH_DIR = os.path.join(BASE_DIR, "image_process_input")  # adjust if needed
PHOTO_TIMEOUT = 30  # seconds max to wait before giving up

def _parse_value(tok):
    try:
        return float(tok.split("=")[1])
    except Exception:
        return None

def load_sequences(seq_path):
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
    if wp["type"] == "cartesian":
        new_data = wp["data"].copy()
        new_data[0] += dx
        new_data[1] += dy
        return {"type": "cartesian", "data": new_data, "grip": wp["grip"]}
    else:
        return wp

def _offset_sequences(base_sequences, dx, dy, extra_dx=0.0, extra_dy=0.0, label=""):
    seqs = {}
    for key, seq in base_sequences.items():
        if key in (0, 1, 2, 3):
            pts = [_apply_offset_to_point(wp, dx + extra_dx, dy + extra_dy) for wp in seq["points"]]
            seqs[key] = {
                "name": seq["name"] + f" (offset {dx+extra_dx:.1f},{dy+extra_dy:.1f})",
                "points": pts
            }
        else:
            seqs[key] = seq
    return seqs

def generate_grid_sequences(base_sequences):
    grid_runs = []
    index = 1
    for row in range(ROWS):
        for col in range(COLS):
            dx = row * X_SPACING
            dy = col * Y_SPACING
            label = f"{chr(ord('A')+row)}{col+1}"
            extra_dx, extra_dy = CUSTOM_OFFSETS.get(label, (0.0, 0.0))
            seqs = _offset_sequences(base_sequences, dx, dy, extra_dx, extra_dy, label)
            grid_runs.append((index, label, seqs))
            index += 1
    return grid_runs

# --- Runner ---
def run_sequences(robot,
                  sequences: dict,
                  run_vector=RUN_VECTOR,
                  move_wait=MOVE_WAIT,
                  photo_prep_seq=PHOTO_PREP_SEQ,
                  photo_prep_wait=PHOTO_PREP_WAIT,
                  photo_seq=PHOTO_SEQ,
                  photo_wait=PHOTO_WAIT,
                  camera_trigger=None,
                  post_photo_script=None,
                  max_positions=None):  # 🆕 added
    """
    Run through vial positions defined by ROWS×COLS grid.
    Use `max_positions` to limit how many vial spots to analyze.
    """
    grid = generate_grid_sequences(sequences)

    # 🆕 Apply position limit if provided
    if max_positions is not None:
        print(f"🆕 Limiting run to first {max_positions} positions (out of {len(grid)})")
        grid = grid[:max_positions]
    else:
        print(f"🆕 No limit applied, running all {len(grid)} positions")

    # === Move to safe at start ===
    if SAFE_SEQ is not None and SAFE_SEQ in sequences:
        print(f"🚦 Moving to SAFE sequence {SAFE_SEQ} before run...")
        for wp in sequences[SAFE_SEQ]["points"]:
            if wp["type"] == "cartesian":
                robot.MoveLin(*wp["data"])
            else:
                robot.MoveJoints(*wp["data"])
            if wp["grip"] == "Open":
                robot.GripperOpen()
            elif wp["grip"] == "Closed":
                robot.GripperClose()
            if move_wait and move_wait > 0:
                time.sleep(move_wait)

    # === Run through all vials ===
    for idx, label, seqs in grid:
        print(f"\n=== ▶ Starting vial position {label} (#{idx}) ===")
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
                if wtype == "cartesian":
                    robot.MoveLin(*data)
                else:
                    robot.MoveJoints(*data)

                if grip == "Open":
                    robot.GripperOpen()
                elif grip == "Closed":
                    robot.GripperClose()

                if move_wait and move_wait > 0:
                    time.sleep(move_wait)

            if key in photo_prep:
                print(f"⚡ [PhotoPrep] Trigger camera at Sequence {key}")
                try:
                    camera_trigger()
                except Exception as e:
                    print(f"⚠️ camera_trigger() failed: {e}")
                if photo_prep_wait and photo_prep_wait > 0:
                    print(f"⏸️ [PhotoPrep] Waiting {photo_prep_wait}s")
                    time.sleep(photo_prep_wait)

            if key in photo_pose:
                print(f"📸 [PhotoPose] Sequence {key} reached — waiting for burst download...")

                # --- Make sure photo directory exists ---
                os.makedirs(WATCH_DIR, exist_ok=True)

                # --- Snapshot current files before waiting ---
                try:
                    before_files = set(os.listdir(WATCH_DIR))
                except Exception as e:
                    print(f"⚠️ Could not list {WATCH_DIR}: {e}")
                    before_files = set()

                # --- Wait for 4 new photos to appear ---
                print(f"📸 Waiting for 4 new photos (timeout {PHOTO_TIMEOUT}s)...")
                start_time = time.time()
                new_files = set()
                while True:
                    try:
                        after_files = set(os.listdir(WATCH_DIR))
                    except Exception as e:
                        print(f"⚠️ Could not read {WATCH_DIR}: {e}")
                        after_files = before_files

                    new_files = after_files - before_files
                    if len(new_files) >= 4:
                        print(f"✅ 4 new photos detected: {', '.join(list(new_files)[:4])}")
                        break

                    if time.time() - start_time > PHOTO_TIMEOUT:
                        print(f"⚠️ Timeout waiting for 4 photos — only {len(new_files)} found.")
                        break
                    time.sleep(0.5)

                # --- Optional hold time after confirmation ---
                if photo_wait and photo_wait > 0:
                    print(f"⏸️ [PhotoPose] Holding {photo_wait}s after 4-photo burst")
                    time.sleep(photo_wait)

                # --- Launch post-analysis script ---
                if post_photo_script:
                    try:
                        script_path = os.path.join(os.path.dirname(__file__), post_photo_script)
                        print(f"▶️ Launching {post_photo_script} for {label} ...")
                        subprocess.Popen([sys.executable, script_path, label])
                    except Exception as e:
                        print(f"⚠️ Failed to launch {post_photo_script}: {e}")



        print(f"=== ✅ Finished vial position {label} ===")

    # === Move to safe at end ===
    if SAFE_SEQ is not None and SAFE_SEQ in sequences:
        print(f"🚦 Moving to SAFE sequence {SAFE_SEQ} after run...")
        for wp in sequences[SAFE_SEQ]["points"]:
            if wp["type"] == "cartesian":
                robot.MoveLin(*wp["data"])
            else:
                robot.MoveJoints(*wp["data"])
            if wp["grip"] == "Open":
                robot.GripperOpen()
            elif wp["grip"] == "Closed":
                robot.GripperClose()
            if move_wait and move_wait > 0:
                time.sleep(move_wait)

    print("\n✅ Selected vial positions complete (robot remains connected).")
