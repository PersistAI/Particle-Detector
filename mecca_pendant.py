import keyboard
import time
import os
from mecademicpy.robot import Robot

# =====================
# CONFIG
ROBOT_IP = "192.168.1.100"   # Replace with your robot's IP
BASE_MM  = 1.0               # Base linear step size (mm)
BASE_DEG = 1.0               # Base angular/joint step size (deg)
DELAY    = 0.1               # Debounce time for jogging
SEQUENCE_FILE = "sequences_dualmode.txt"
# =====================


def print_state_cart(pose, gripper_state, speed_scale=None):
    labels = ["X", "Y", "Z", "α", "β", "γ"]
    vec = ", ".join([f"{labels[i]}={pose[i]:.2f}" for i in range(6)])
    if speed_scale is not None:
        return f"[{vec}, Gripper={gripper_state}, SpeedScale={speed_scale:.2f}]"
    return f"[{vec}, Gripper={gripper_state}]"


def print_state_joints(joints, gripper_state, speed_scale=None):
    vec = ", ".join([f"J{i+1}={joints[i]:.2f}" for i in range(6)])
    if speed_scale is not None:
        return f"[{vec}, Gripper={gripper_state}, SpeedScale={speed_scale:.2f}]"
    return f"[{vec}, Gripper={gripper_state}]"


def save_sequences(sequences):
    with open(SEQUENCE_FILE, "w", encoding="utf-8") as f:
        for k, v in sequences.items():
            f.write(f"==== SEQUENCE {k} ({v['name']}) ====\n")
            for wp in v["points"]:
                wtype = wp["type"]
                grip  = wp["grip"]
                if wtype == "cartesian":
                    X,Y,Z,a,b,g = wp["data"]
                    f.write(f"TYPE=cartesian, X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}, α={a:.3f}, β={b:.3f}, γ={g:.3f}, Gripper={grip}\n")
                else:
                    j = wp["data"]
                    f.write(f"TYPE=joints, J1={j[0]:.3f}, J2={j[1]:.3f}, J3={j[2]:.3f}, J4={j[3]:.3f}, J5={j[4]:.3f}, J6={j[5]:.3f}, Gripper={grip}\n")
            f.write("\n")
    print(f"Sequences saved to {SEQUENCE_FILE}")


def _parse_value(tok):
    try:
        return float(tok.split("=")[1])
    except Exception:
        return None


def load_sequences():
    sequences = {}
    if not os.path.exists(SEQUENCE_FILE):
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

    print(f"Loaded {len(sequences)} sequences from {SEQUENCE_FILE}")
    return sequences


def get_valid_pose(robot, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        pose = robot.GetPose()
        if any(abs(v) > 1e-3 for v in pose):
            return pose
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for valid robot pose")


def execute_sequence_step(robot, seq, key):
    print(f"Step mode: Press NumPad {key} to move through {seq['name']}")
    idx = 0
    while idx < len(seq["points"]):
        if keyboard.is_pressed(f"num {key}"):
            wp = seq["points"][idx]
            wtype, data, grip = wp["type"], wp["data"], wp["grip"]

            try:
                if wtype == "cartesian":
                    robot.MoveLin(*data)
                else:
                    robot.MoveJoints(*data)
            except Exception as e:
                print(f"⚠️ Move rejected during sequence: {e}")

            if grip == "Open":
                robot.GripperOpen()
            elif grip == "Closed":
                robot.GripperClose()

            if wtype == "cartesian":
                print(print_state_cart(data, grip))
            else:
                print(print_state_joints(data, grip))

            idx += 1
            time.sleep(0.5)
    print(f"Sequence {key} complete.\n")


def main():
    robot = Robot()
    robot.Connect(ROBOT_IP)
    robot.WaitConnected()

    print("Connected to Mecademic robot.")
    robot.ActivateRobot()
    robot.Home()
    robot.WaitHomed()

    mode = "joints"
    current_joints = [0.0]*6
    current_pose   = get_valid_pose(robot)
    gripper_state  = "Unknown"
    speed_scale    = 1.0
    sequences      = load_sequences()
    current_sequence = []

    try:
        print("Moving to ABSOLUTE joint zero [0,0,0,0,0,0] ...")
        robot.MoveJoints(*current_joints)
    except Exception as e:
        print(f"⚠️ Could not move to joint zero: {e}")
        current_joints = robot.GetJoints()

    print("\n=== Controls ===")
    print("Mode toggle: Spacebar  (cartesian ⇄ joints)")
    print("Reset: Backspace (ResetError + ResumeMotion + resync)")
    print("Speed: + increase / - decrease (scales XYZ & joints/angles)")
    print("Gripper: O=open, P=close")
    print("\nCARTESIAN mode jogging: W/S=X, A/D=Y, Q/E=Z, H/K=α, U/J=β, I/Y=γ")
    print("\nJOINTS mode jogging:")
    print("  J1: A(+), D(-)")
    print("  J2: S(+), W(-)")
    print("  J3: Shift+S(+), Shift+W(-)")
    print("  J4: E(+), Q(-)")
    print("  J5: Shift+Alt+S(+), Shift+Alt+W(-)")
    print("  J6: Shift+E(+), Shift+Q(-)")
    print("================\n")

    try:
        while True:
            moved = False

            # --- Reset ---
            if keyboard.is_pressed("backspace"):
                print("🔄 Resetting robot error and resuming motion...")
                robot.ResetError()
                robot.ResumeMotion()
                current_pose   = get_valid_pose(robot)
                current_joints = robot.GetJoints()
                time.sleep(0.5)

            # --- Mode toggle ---
            if keyboard.is_pressed("space"):
                if mode == "cartesian":
                    mode = "joints"
                    current_joints = robot.GetJoints()
                    print("🔁 Switched to JOINTS mode:", print_state_joints(current_joints, gripper_state, speed_scale))
                else:
                    mode = "cartesian"
                    current_pose = get_valid_pose(robot)
                    print("🔁 Switched to CARTESIAN mode:", print_state_cart(current_pose, gripper_state, speed_scale))
                time.sleep(0.3)

            # --- Speed scale ---
            if keyboard.is_pressed("+"):
                if speed_scale < 5: speed_scale += 0.1
                elif speed_scale < 50: speed_scale += 1
                else: speed_scale += 5
                print(f"Speed scale → {speed_scale:.2f}")
                time.sleep(0.3)
            if keyboard.is_pressed("-"):
                if speed_scale > 50: speed_scale -= 5
                elif speed_scale > 5: speed_scale -= 1
                else: speed_scale = max(0.1, speed_scale - 0.1)
                print(f"Speed scale → {speed_scale:.2f}")
                time.sleep(0.3)

            step_mm  = BASE_MM  * speed_scale
            step_deg = BASE_DEG * speed_scale

            # --- Jogging ---
            if mode == "cartesian":
                if keyboard.is_pressed("w"): current_pose[0]+=step_mm; moved=True
                if keyboard.is_pressed("s"): current_pose[0]-=step_mm; moved=True
                if keyboard.is_pressed("a"): current_pose[1]+=step_mm; moved=True
                if keyboard.is_pressed("d"): current_pose[1]-=step_mm; moved=True
                if keyboard.is_pressed("q"): current_pose[2]+=step_mm; moved=True
                if keyboard.is_pressed("e"): current_pose[2]-=step_mm; moved=True
                if keyboard.is_pressed("h"): current_pose[3]+=step_deg; moved=True
                if keyboard.is_pressed("k"): current_pose[3]-=step_deg; moved=True
                if keyboard.is_pressed("u"): current_pose[4]+=step_deg; moved=True
                if keyboard.is_pressed("j"): current_pose[4]-=step_deg; moved=True
                if keyboard.is_pressed("i"): current_pose[5]+=step_deg; moved=True
                if keyboard.is_pressed("y"): current_pose[5]-=step_deg; moved=True
                if moved:
                    try: robot.MoveLin(*current_pose)
                    except Exception as e: print(f"⚠️ Move rejected: {e}")
                    print(print_state_cart(current_pose, gripper_state, speed_scale))
                    time.sleep(DELAY)
            else:
                j = current_joints
                if keyboard.is_pressed("a"): j[0]+=step_deg; moved=True
                if keyboard.is_pressed("d"): j[0]-=step_deg; moved=True
                # --- J5 check first (Shift+Alt+W/S) ---
                if keyboard.is_pressed("shift") and keyboard.is_pressed("alt"):
                    if keyboard.is_pressed("s"): j[4]+=step_deg; moved=True
                    if keyboard.is_pressed("w"): j[4]-=step_deg; moved=True
                # J2/J3 (only if NOT Alt)
                elif keyboard.is_pressed("s"):
                    if keyboard.is_pressed("shift"): j[2]+=step_deg
                    else: j[1]+=step_deg
                    moved=True
                elif keyboard.is_pressed("w"):
                    if keyboard.is_pressed("shift"): j[2]-=step_deg
                    else: j[1]-=step_deg
                    moved=True
                # J4/J6
                if keyboard.is_pressed("e"):
                    if keyboard.is_pressed("shift"): j[5]+=step_deg
                    else: j[3]+=step_deg
                    moved=True
                if keyboard.is_pressed("q"):
                    if keyboard.is_pressed("shift"): j[5]-=step_deg
                    else: j[3]-=step_deg
                    moved=True
                if moved:
                    try: robot.MoveJoints(*j)
                    except Exception as e: print(f"⚠️ Move rejected: {e}")
                    print(print_state_joints(j, gripper_state, speed_scale))
                    time.sleep(DELAY)

            # --- Gripper ---
            if keyboard.is_pressed("o"):
                gripper_state="Open"; robot.GripperOpen()
                print("Opening gripper")
                time.sleep(0.3)
            if keyboard.is_pressed("p"):
                gripper_state="Closed"; robot.GripperClose()
                print("Closing gripper")
                time.sleep(0.3)

            # --- Teach ---
            if keyboard.is_pressed("num enter"):
                if mode=="cartesian":
                    pose=get_valid_pose(robot)
                    current_sequence.append({"type":"cartesian","data":pose,"grip":gripper_state})
                    print("Recorded CARTESIAN:", print_state_cart(pose, gripper_state))
                else:
                    joints=robot.GetJoints()
                    current_sequence.append({"type":"joints","data":joints,"grip":gripper_state})
                    print("Recorded JOINTS:", print_state_joints(joints, gripper_state))
                time.sleep(0.3)

            # --- Assign sequence ---
            for key in range(10):
                if keyboard.is_pressed(f"num {key}") and current_sequence:
                    seq_name=f"Sequence {key}"
                    sequences[key]={"name":seq_name,"points":list(current_sequence)}
                    print(f"Sequence {key} saved with {len(current_sequence)} points.")
                    save_sequences(sequences)
                    current_sequence.clear()
                    time.sleep(0.5)
                    break

            # --- Execute sequence ---
            for key in range(10):
                if keyboard.is_pressed(f"num {key}") and key in sequences and not current_sequence:
                    execute_sequence_step(robot,sequences[key],key)
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print("Ctrl+C detected, stopping...")
    finally:
        robot.DeactivateRobot()
        robot.Disconnect()
        save_sequences(sequences)
        print("Robot disconnected. Bye!")


if __name__ == "__main__":
    main()
