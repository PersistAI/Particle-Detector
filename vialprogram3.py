import cv2
import numpy as np
import glob
import os
from collections import defaultdict
import re
from datetime import datetime
import sys
import json
import time

vial_label = sys.argv[1] if len(sys.argv) > 1 else "Unknown"

# ==============================
# USER SETTINGS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to script location
INPUT_PATH = os.path.join(BASE_DIR, "image_process_input", "*.[jJ][pP][gG]")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "image_process_output")

# Detection Parameters
X_MIN, X_MAX = 1570, 5790
Y_MIN, Y_MAX = 1980, 3632
BRIGHTNESS_PERCENTILE = 96.66080167002514
MIN_BLOB_AREA = 94603
MAX_BLOB_AREA = 4598289
BLUR_SIZE = 51
MORPH_KERNEL_SIZE = 37

# ==============================
def natural_sort_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

def load_images(file_list):
    images = []
    valid_files = []
    print(f"  Loading {len(file_list)} image(s)...")
    for f in file_list:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            valid_files.append(f)
            print(f"    ✓ Loaded: {os.path.basename(f)} ({img.shape[1]}x{img.shape[0]})")
        else:
            print(f"    ✗ Warning: Could not load {f}")
    return valid_files, images

def preprocess_image(image):
    """Crop and enhance image"""
    print("  Preprocessing image...")
    print(f"    Original size: {image.shape[1]}x{image.shape[0]}")
    
    # Crop to ROI
    roi = image[Y_MIN:Y_MAX, X_MIN:X_MAX]
    print(f"    ROI size: {roi.shape[1]}x{roi.shape[0]}")
    
    # Smooth the image to reduce noise
    blurred = cv2.GaussianBlur(roi, (BLUR_SIZE, BLUR_SIZE), 0)
    
    print("  ✓ Preprocessing complete")
    return blurred

def detect_oil_droplets_simple(image):
    """Detect brightest regions (oil droplets) using percentile thresholding"""
    print("  Detecting brightest regions...")
    print(f"    Brightness percentile: {BRIGHTNESS_PERCENTILE}%")
    print(f"    Min area: {MIN_BLOB_AREA}px², Max area: {MAX_BLOB_AREA}px²")
    
    start_time = time.time()
    
    # Find brightness threshold based on percentile
    threshold_value = np.percentile(image, BRIGHTNESS_PERCENTILE)
    print(f"    Calculated threshold: {threshold_value:.0f}/255")
    
    # Threshold to get brightest regions
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    print(f"    Bright pixels found: {np.sum(binary > 0)}")
    
    # Morphological operations to clean up and connect nearby bright regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Close small gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    print(f"  ✓ Found {num_labels - 1} connected components")
    
    detected_droplets = []
    
    # Iterate through components (skip background label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by area
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            # Get centroid
            cx, cy = centroids[i]
            cx, cy = int(cx), int(cy)
            
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Create mask for this component
            mask = (labels == i).astype(np.uint8) * 255
            
            # Find contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = contours[0]
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Convert back to original image coordinates
                orig_x = cx + X_MIN
                orig_y = cy + Y_MIN
                
                # Calculate equivalent radius
                radius = int(np.sqrt(area / np.pi))
                
                detected_droplets.append({
                    'center': (orig_x, orig_y),
                    'radius': radius,
                    'area': float(area),
                    'circularity': circularity,
                    'contour': contour
                })
                
                print(f"    Droplet {len(detected_droplets)}: center=({orig_x},{orig_y}), "
                      f"area={area:.0f}px², radius={radius}px, circularity={circularity:.2f}")
    
    elapsed = time.time() - start_time
    print(f"  ✓ Detection complete in {elapsed:.1f}s")
    print(f"  ✓ {len(detected_droplets)} droplets detected after filtering")
    
    return detected_droplets

def detect_droplets_multi_frame(images):
    """Detect droplets across multiple frames and aggregate results"""
    all_detections = []
    
    for i, img in enumerate(images):
        print(f"\n  Processing Frame {i+1}/{len(images)}:")
        preprocessed = preprocess_image(img)
        droplets = detect_oil_droplets_simple(preprocessed)
        
        for droplet in droplets:
            droplet['frame'] = i
            all_detections.append(droplet)
        
        print(f"  Frame {i+1}/{len(images)}: {len(droplets)} droplets detected")
    
    # Cluster detections across frames (same droplet detected multiple times)
    print("\n  Clustering detections across frames...")
    clustered_droplets = cluster_droplet_detections(all_detections)
    print(f"  ✓ Clustered into {len(clustered_droplets)} unique droplets")
    
    return clustered_droplets, all_detections

def cluster_droplet_detections(detections, distance_threshold=200):
    """Group detections that are likely the same droplet across frames"""
    if not detections:
        return []
    
    clusters = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        
        cluster = [det1]
        used.add(i)
        
        for j, det2 in enumerate(detections):
            if j in used or j <= i:
                continue
            
            # Calculate distance between centers
            dx = det1['center'][0] - det2['center'][0]
            dy = det1['center'][1] - det2['center'][1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < distance_threshold:
                cluster.append(det2)
                used.add(j)
        
        # Average the cluster properties
        avg_x = int(np.mean([d['center'][0] for d in cluster]))
        avg_y = int(np.mean([d['center'][1] for d in cluster]))
        avg_radius = int(np.mean([d['radius'] for d in cluster]))
        avg_area = np.mean([d['area'] for d in cluster])
        avg_circularity = np.mean([d['circularity'] for d in cluster])
        frames_seen = [d['frame'] for d in cluster]
        
        clusters.append({
            'center': (avg_x, avg_y),
            'radius': avg_radius,
            'area': avg_area,
            'circularity': avg_circularity,
            'frames': frames_seen,
            'detection_count': len(cluster)
        })
    
    # Sort by area (largest first)
    clusters.sort(key=lambda x: x['area'], reverse=True)
    
    return clusters

def create_marked_frames(images, all_detections):
    """Create individual frames with detected droplets marked"""
    print("\n  Creating marked frames...")
    marked_frames = []
    
    for i, img in enumerate(images):
        marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw ROI
        cv2.rectangle(marked, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 255, 0), 2)
        
        # Draw droplets detected in this frame
        frame_detections = [d for d in all_detections if d['frame'] == i]
        
        for droplet in frame_detections:
            center = droplet['center']
            radius = droplet['radius']
            
            # Draw circle
            cv2.circle(marked, center, radius, (0, 0, 255), 4)
            # Draw center point
            cv2.circle(marked, center, 8, (255, 0, 0), -1)
            
            # Add label
            label = f"{int(droplet['area'])}px²"
            cv2.putText(marked, label, 
                       (center[0] + radius + 10, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Frame info
        cv2.putText(marked, f"Frame {i+1}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(marked, f"Droplets: {len(frame_detections)}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        marked_frames.append(marked)
        print(f"    Frame {i+1}: marked with {len(frame_detections)} droplets")
    
    return marked_frames

def create_result_visualization(base_image, clustered_droplets):
    """Create summary visualization with all detected droplets"""
    print("\n  Creating result visualization...")
    result = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    
    # Draw ROI
    cv2.rectangle(result, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 255, 0), 3)
    
    # Draw each detected droplet
    for i, droplet in enumerate(clustered_droplets):
        center = droplet['center']
        radius = droplet['radius']
        
        # Draw circle
        cv2.circle(result, center, radius, (0, 0, 255), 5)
        # Draw center
        cv2.circle(result, center, 10, (255, 0, 0), -1)
        
        # Add label with number and area
        label = f"#{i+1}: {int(droplet['area'])}px²"
        cv2.putText(result, label, 
                   (center[0] + radius + 15, center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Legend
    legend_y = 60
    cv2.putText(result, "Legend:", (20, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.circle(result, (40, legend_y + 35), 20, (0, 0, 255), 5)
    cv2.putText(result, "Oil Droplet", (75, legend_y + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Summary text
    droplet_count = len(clustered_droplets)
    result_text = f"DROPLETS DETECTED: {droplet_count}" if droplet_count > 0 else "NO DROPLETS DETECTED"
    color = (0, 0, 255) if droplet_count > 0 else (0, 255, 0)
    cv2.putText(result, result_text, (X_MIN, Y_MIN - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    
    print("  ✓ Visualization complete")
    return result

def analyze_vial_sequence(vial_files):
    """Analyze all images for oil droplets"""
    print(f"\nAnalyzing vial with {len(vial_files)} images:")
    for f in vial_files:
        print(f"  - {os.path.basename(f)}")
    
    valid_files, images = load_images(vial_files)
    
    if len(images) == 0:
        print("  ✗ Error: No valid images loaded")
        return False, None, None
    
    print(f"\nDetecting oil droplets...")
    clustered_droplets, all_detections = detect_droplets_multi_frame(images)
    
    print(f"\n  Total unique droplets: {len(clustered_droplets)}")
    print(f"  Total detections across frames: {len(all_detections)}")
    
    # Use the last image as base for result visualization
    result_image = create_result_visualization(images[-1], clustered_droplets)
    
    has_droplets = len(clustered_droplets) > 0
    
    return has_droplets, result_image, {
        'droplet_count': len(clustered_droplets),
        'total_detections': len(all_detections),
        'droplets': clustered_droplets,
        'all_detections': all_detections,
        'valid_files': valid_files,
        'images': images
    }

def main():
    print(f"Script started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Vial label: {vial_label}")
    print(f"Looking for images in: {INPUT_PATH}")
    
    all_files = sorted(glob.glob(INPUT_PATH), key=natural_sort_key)
    
    if len(all_files) == 0:
        print("✗ No images found. Check INPUT_PATH.")
        print(f"  Searched: {INPUT_PATH}")
        print(f"  Directory exists: {os.path.exists(os.path.dirname(INPUT_PATH))}")
        return
    
    print(f"Found {len(all_files)} images in sequence:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ANALYZING VIAL: {vial_label}")
    print(f"{'='*60}")
    
    has_droplets, result_image, analysis = analyze_vial_sequence(all_files)
    
    if has_droplets:
        status = "✅ DROPLETS DETECTED"
        print(f"\n{status}")
        print(f"Oil droplets found: {analysis['droplet_count']}")
        
        for i, droplet in enumerate(analysis['droplets']):
            radius = droplet['radius']
            area = droplet['area']
            circ = droplet['circularity']
            frames = len(droplet['frames'])
            conf = droplet['detection_count']
            print(f"  Droplet {i+1}: radius={radius}px, area={area:.0f}px², "
                  f"circularity={circ:.2f}, seen in {frames} frames, confidence={conf}")
    else:
        status = "❌ NO DROPLETS DETECTED"
        print(f"\n{status}")
    
    # Save main result
    if result_image is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_result_filename = f"{vial_label}_RESULT_{timestamp}.jpg"
        main_result_path = os.path.join(OUTPUT_FOLDER, main_result_filename)
        cv2.imwrite(main_result_path, result_image)
        print(f"\n✓ Saved main result: {main_result_filename}")
    
    # Save individual marked frames
    if len(analysis['images']) > 0:
        marked_frames = create_marked_frames(analysis['images'], analysis['all_detections'])
        
        print(f"\nSaving individual frame markings:")
        for i, marked_frame in enumerate(marked_frames):
            frame_filename = f"{vial_label}_Frame{i+1:02d}.jpg"
            frame_path = os.path.join(OUTPUT_FOLDER, frame_filename)
            cv2.imwrite(frame_path, marked_frame)
            print(f"  ✓ Frame {i+1}: {frame_filename}")
    
    # Save JSON log
    log_filename = "droplet_detection_log.txt"
    log_path = os.path.join(OUTPUT_FOLDER, log_filename)
    
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vial_id": vial_label,
        "images_analyzed": len(all_files),
        "result": status.replace("✅ ", "").replace("❌ ", ""),
        "droplets_detected": analysis['droplet_count'],
        "total_detections": analysis['total_detections'],
        "output_main": main_result_filename,
        "droplet_details": [
            {
                "droplet_num": i+1,
                "radius_px": d['radius'],
                "area_px2": round(d['area'], 1),
                "circularity": round(d['circularity'], 2),
                "frames_seen": len(d['frames']),
                "confidence": d['detection_count']
            }
            for i, d in enumerate(analysis['droplets'])
        ]
    }
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        json.dump(entry, log_file)
        log_file.write("\n")
    
    print(f"\n✓ JSON log entry added to: {log_filename}")
    
    # Clean up input files
    print("\nDeleting input images...")
    for f in all_files:
        try:
            os.remove(f)
            print(f"  ✓ Deleted: {os.path.basename(f)}")
        except Exception as e:
            print(f"  ✗ Could not delete {f}: {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Images processed: {len(all_files)}")
    print(f"Oil droplets detected: {analysis['droplet_count']}")
    print(f"All results saved to: {OUTPUT_FOLDER}")
    print(f"Check the log file for historical records: {log_filename}")

if __name__ == "__main__":
    main()