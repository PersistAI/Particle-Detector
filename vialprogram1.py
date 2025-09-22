import cv2
import numpy as np
import glob
import os
from collections import defaultdict
import re
import shutil

# ==============================
# USER SETTINGS
# ==============================
# Base directory = folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to script location
INPUT_PATH = os.path.join(BASE_DIR, "image_process_input", "*.jpg")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "image_process_output")
ARCHIVE_FOLDER = os.path.join(BASE_DIR, "image_process_archive")

#Sensitivity values
MIN_AREA = 3                    # Minimum blob area in pixels
MAX_AREA = 20
THRESH_VALUE = 30               # Threshold sensitivity (lower = more sensitive)
# Vial ROI (rectangle) - adjust these numbers for your setup
X_MIN, X_MAX = 1970, 5390       # horizontal crop (left,right)
Y_MIN, Y_MAX = 2280, 3632       # vertical crop (top,bottom)
#===============================
# Motion detection settings
MIN_MOVEMENT_DISTANCE = 5       # Minimum pixel distance to consider as movement
MAX_TRACKING_DISTANCE = 50      # Maximum distance to track same particle between frames
MOTION_SENSITIVITY = 10         # Sensitivity for background subtraction (lower = more sensitive)
MIN_FRAMES_SEEN = 2             # Minimum frames a particle must be seen in to be considered real
STATIC_REMOVAL_THRESHOLD = 3    # Remove particles that appear in same location across this many frames
# ==============================

def natural_sort_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

def group_images_by_vial(files):
    vial_groups = defaultdict(list)
    for file in files:
        basename = os.path.basename(file)
        if basename.startswith('DCS_'):
            vial_id = 'DCS'
        else:
            parts = re.split(r'_(\d+)', basename)
            if len(parts) > 1:
                vial_id = parts[0]
            else:
                vial_id = basename.split('.')[0]
        vial_groups[vial_id].append(file)
    for vial_id in vial_groups:
        vial_groups[vial_id].sort(key=natural_sort_key)
    return dict(vial_groups)

def load_images(file_list):
    images = []
    valid_files = []
    for f in file_list:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            valid_files.append(f)
        else:
            print(f"Warning: Could not load {f}")
    return valid_files, images

def preprocess_image(image):
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[Y_MIN:Y_MAX, X_MIN:X_MAX] = 255
    roi = cv2.bitwise_and(image, image, mask=mask)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)
    return enhanced

def detect_static_particles(images):
    if len(images) < 2:
        return []
    background = np.median(np.array(images), axis=0).astype(np.uint8)
    static_particles = []
    _, thresh = cv2.threshold(background, THRESH_VALUE, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA <= area <= MAX_AREA:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                static_particles.append((cx, cy, area))
    return static_particles

def detect_moving_particles(images):
    if len(images) < 2:
        return [], []
    moving_particles = []
    all_detections = []
    frame_particles = []
    for i, img in enumerate(images):
        _, thresh = cv2.threshold(img, THRESH_VALUE, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= MIN_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if (X_MIN + 10 < cx < X_MAX - 10 and Y_MIN + 10 < cy < Y_MAX - 10):
                        frame_detections.append({
                            'center': (cx, cy),
                            'area': area,
                            'frame': i,
                            'contour': contour
                        })
        frame_particles.append(frame_detections)
        all_detections.extend(frame_detections)
    if len(frame_particles) >= 2:
        moving_particles = track_particles_across_frames(frame_particles)
    return moving_particles, all_detections

def track_particles_across_frames(frame_particles):
    moving_particles = []
    if len(frame_particles) < 2:
        return moving_particles
    max_particles_per_frame = 200
    for i, frame in enumerate(frame_particles):
        if len(frame) > max_particles_per_frame:
            print(f"  Warning: Frame {i+1} has {len(frame)} detections (likely noise).")
            frame.sort(key=lambda p: p['area'], reverse=True)
            frame_particles[i] = frame[:max_particles_per_frame]
    tracked_particles = []
    for particle in frame_particles[0]:
        tracked_particles.append({
            'positions': [particle['center']],
            'areas': [particle['area']],
            'frames': [0],
            'last_seen': 0,
            'total_movement': 0
        })
    total_frames = len(frame_particles)
    for frame_idx in range(1, total_frames):
        print(f"    Tracking frame {frame_idx+1}/{total_frames}...", end='\r')
        current_particles = frame_particles[frame_idx]
        for curr_particle in current_particles:
            curr_pos = curr_particle['center']
            best_match = None
            min_distance = float('inf')
            active_tracked = [t for t in tracked_particles if t['last_seen'] >= frame_idx - 2]
            for tracked in active_tracked:
                last_pos = tracked['positions'][-1]
                dx = curr_pos[0] - last_pos[0]
                dy = curr_pos[1] - last_pos[1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance < MAX_TRACKING_DISTANCE and distance < min_distance:
                    min_distance = distance
                    best_match = tracked
            if best_match:
                best_match['positions'].append(curr_pos)
                best_match['areas'].append(curr_particle['area'])
                best_match['frames'].append(frame_idx)
                best_match['last_seen'] = frame_idx
                best_match['total_movement'] += min_distance
            else:
                if len(tracked_particles) < 1000:
                    tracked_particles.append({
                        'positions': [curr_pos],
                        'areas': [curr_particle['area']],
                        'frames': [frame_idx],
                        'last_seen': frame_idx,
                        'total_movement': 0
                    })
    print("    Filtering for moving particles...                    ")
    for tracked in tracked_particles:
        if (len(tracked['frames']) >= MIN_FRAMES_SEEN and 
            tracked['total_movement'] >= MIN_MOVEMENT_DISTANCE):
            moving_particles.append(tracked)
    return moving_particles

def create_individual_frame_markings(images, static_particles, moving_particles):
    marked_frames = []
    for i, img in enumerate(images):
        marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(marked, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 255, 0), 2)
        for (x, y, area) in static_particles:
            cv2.circle(marked, (x, y), 4, (255, 100, 0), 1)
        particles_in_frame = 0
        for particle in moving_particles:
            if i in particle['frames']:
                frame_index = particle['frames'].index(i)
                pos = particle['positions'][frame_index]
                cv2.circle(marked, pos, 6, (0, 0, 255), 2)
                particles_in_frame += 1
        cv2.putText(marked, f"Frame {i+1}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(marked, f"Moving particles: {particles_in_frame}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        marked_frames.append(marked)
    return marked_frames

def analyze_vial_sequence(vial_files):
    print(f"\nAnalyzing vial with {len(vial_files)} images:")
    for f in vial_files:
        print(f"  - {os.path.basename(f)}")
    valid_files, images = load_images(vial_files)
    if len(images) < 2:
        print("  Warning: Need at least 2 images for motion detection")
        return False, None, None
    processed_images = [preprocess_image(img) for img in images]
    static_particles = detect_static_particles(processed_images)
    print(f"  Static particles detected: {len(static_particles)}")
    moving_particles, all_detections = detect_moving_particles(processed_images)
    print(f"  Moving particles detected: {len(moving_particles)}")
    result_image = create_result_visualization(images[-1], static_particles, moving_particles)
    has_particles = len(moving_particles) > 0
    return has_particles, result_image, {
        'static_count': len(static_particles),
        'moving_count': len(moving_particles),
        'total_detections': len(all_detections),
        'moving_particles': moving_particles
    }

def create_result_visualization(base_image, static_particles, moving_particles):
    result = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 255, 0), 2)
    for (x, y, area) in static_particles:
        cv2.circle(result, (x, y), 4, (255, 100, 0), 1)
    for particle in moving_particles:
        positions = particle['positions']
        for i in range(len(positions) - 1):
            cv2.line(result, positions[i], positions[i+1], (0, 255, 255), 1)
        final_pos = positions[-1]
        cv2.circle(result, final_pos, 6, (0, 0, 255), 2)
        movement = particle['total_movement']
        cv2.putText(result, f"{movement:.1f}", 
                   (final_pos[0] + 10, final_pos[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    legend_y = 50
    cv2.putText(result, "Legend:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.circle(result, (30, legend_y + 25), 4, (255, 100, 0), -1)
    cv2.putText(result, "Static (artifacts)", (50, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(result, (30, legend_y + 50), 6, (0, 0, 255), 2)
    cv2.putText(result, "Moving (particles)", (50, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    particle_count = len(moving_particles)
    result_text = f"PARTICLES DETECTED: {particle_count}" if particle_count > 0 else "NO PARTICLES DETECTED"
    color = (0, 0, 255) if particle_count > 0 else (0, 255, 0)
    cv2.putText(result, result_text, (X_MIN, Y_MIN - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return result

def main():
    all_files = sorted(glob.glob(INPUT_PATH), key=natural_sort_key)
    if len(all_files) == 0:
        print("No images found. Check INPUT_PATH.")
        return
    print(f"Found {len(all_files)} images in sequence:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    vial_groups = group_images_by_vial(all_files)
    vial_id = list(vial_groups.keys())[0]
    files = vial_groups[vial_id]
    print(f"\n{'='*60}")
    print(f"ANALYZING VIAL SEQUENCE: {vial_id}")
    print(f"{'='*60}")
    has_particles, result_image, analysis = analyze_vial_sequence(files)
    if has_particles:
        status = "✅ PARTICLES DETECTED"
        print(f"\n{status}")
        print(f"Moving particles found: {analysis['moving_count']}")
        for i, particle in enumerate(analysis['moving_particles']):
            frames_seen = len(particle['frames'])
            total_movement = particle['total_movement']
            avg_area = np.mean(particle['areas'])
            print(f"  Particle {i+1}: {total_movement:.1f}px movement, {frames_seen} frames, {avg_area:.1f}px area")
    else:
        status = "❌ NO PARTICLES DETECTED"
        print(f"\n{status}")
        print(f"Static artifacts found: {analysis['static_count']}")
    if result_image is not None:
        main_result_filename = f"{vial_id}_RESULT.jpg"
        main_result_path = os.path.join(OUTPUT_FOLDER, main_result_filename)
        cv2.imwrite(main_result_path, result_image)
        print(f"\nSaved main result: {main_result_filename}")
    valid_files, images = load_images(files)
    if len(images) >= 2:
        processed_images = [preprocess_image(img) for img in images]
        static_particles = detect_static_particles(processed_images)
        moving_particles = analysis['moving_particles']
        marked_frames = create_individual_frame_markings(images, static_particles, moving_particles)
        print(f"\nSaving individual frame markings:")
        for i, (marked_frame, original_file) in enumerate(zip(marked_frames, valid_files)):
            original_name = os.path.splitext(os.path.basename(original_file))[0]
            frame_filename = f"{vial_id}_Frame{i+1:02d}_{original_name}_marked.jpg"
            frame_path = os.path.join(OUTPUT_FOLDER, frame_filename)
            cv2.imwrite(frame_path, marked_frame)
            print(f"  Frame {i+1}: {frame_filename}")
    log_filename = "particle_detection_log.txt"
    log_path = os.path.join(OUTPUT_FOLDER, log_filename)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
{'='*80}
Analysis Date: {timestamp}
Vial ID: {vial_id}
Images Analyzed: {len(files)}
Result: {status}
Moving Particles: {analysis['moving_count']}
Static Artifacts: {analysis['static_count']}
Files Processed:
"""
    for f in files:
        log_entry += f"  - {os.path.basename(f)}\n"
    if has_particles:
        log_entry += "\nParticle Details:\n"
        for i, particle in enumerate(analysis['moving_particles']):
            frames_seen = len(particle['frames'])
            total_movement = particle['total_movement']
            avg_area = np.mean(particle['areas'])
            log_entry += f"  Particle {i+1}: Movement={total_movement:.1f}px, Frames={frames_seen}, AvgArea={avg_area:.1f}px\n"
    log_entry += f"Output Files Generated:\n"
    log_entry += f"  - {main_result_filename} (summary with trajectories)\n"
    for i, original_file in enumerate(valid_files):
        original_name = os.path.splitext(os.path.basename(original_file))[0]
        frame_filename = f"{vial_id}_{original_name}_marked.jpg"
        log_entry += f"  - {frame_filename}\n"
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(log_entry)
    print(f"\nLog entry added to: {log_filename}")
    # ✅ Move input images to archive
    print("\nArchiving input images...")
    for f in files:
        try:
            dest = os.path.join(ARCHIVE_FOLDER, os.path.basename(f))
            shutil.move(f, dest)
            print(f"  Moved: {os.path.basename(f)}")
        except Exception as e:
            print(f"  Could not move {f}: {e}")
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Images processed: {len(files)}")
    print(f"Moving particles detected: {analysis['moving_count']}")
    print(f"Static artifacts detected: {analysis['static_count']}")
    print(f"All results saved to: {OUTPUT_FOLDER}")
    print(f"Raw images archived to: {ARCHIVE_FOLDER}")
    print(f"Check the log file for historical records: {log_filename}")

if __name__ == "__main__":
    main()