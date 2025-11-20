import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import glob
import os
import threading
from skopt import gp_minimize
from skopt.space import Real, Integer

class DropletDetectorTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Oil Droplet Detection Parameter Tuner with Bayesian Optimization")
        self.root.geometry("1800x900")
        
        # Current images and detection results
        self.current_display_image = None
        self.loaded_images = []  # List of (filename, image) tuples
        self.current_image_index = 0
        self.processing = False
        self.optimizing = False
        self.optimization_thread = None
        
        # Create main layout
        self.create_widgets()
        
        # Default parameters
        self.reset_to_defaults()
        
    def create_widgets(self):
        # Left panel - Parameters
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Make left panel scrollable - INCREASED WIDTH
        canvas = tk.Canvas(left_frame, width=550)  # CHANGED from 400 to 550
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure column weights to ensure proper spacing
        scrollable_frame.columnconfigure(2, weight=1)  # Make slider column expandable
        
        # Script selection
        ttk.Label(scrollable_frame, text="Detection Script:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.script_path_var = tk.StringVar(value="particle_detection.py")
        script_entry = ttk.Entry(scrollable_frame, textvariable=self.script_path_var, width=25)
        script_entry.grid(row=0, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(scrollable_frame, text="Browse", command=self.browse_script).grid(row=0, column=4, pady=5)
        
        # Image selection
        ttk.Label(scrollable_frame, text="Input Folder:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_folder_var = tk.StringVar(value="image_process_input")
        folder_entry = ttk.Entry(scrollable_frame, textvariable=self.input_folder_var, width=25)
        folder_entry.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(scrollable_frame, text="Browse", command=self.browse_folder).grid(row=1, column=4, pady=5)
        
        # Load images button and controls
        image_load_frame = ttk.Frame(scrollable_frame)
        image_load_frame.grid(row=2, column=0, columnspan=5, pady=10)
        
        ttk.Button(image_load_frame, text="📁 Load All Images", command=self.load_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(image_load_frame, text="◀", command=self.prev_image, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(image_load_frame, text="▶", command=self.next_image, width=3).pack(side=tk.LEFT, padx=2)
        
        # Image counter
        self.image_counter_var = tk.StringVar(value="No images loaded")
        ttk.Label(scrollable_frame, textvariable=self.image_counter_var, foreground="blue").grid(row=3, column=0, columnspan=5, sticky=tk.W)
        
        # Loaded images list
        ttk.Label(scrollable_frame, text="Loaded Images & Targets:", font=('Arial', 9, 'bold')).grid(row=4, column=0, columnspan=5, sticky=tk.W, pady=(10,5))
        
        # Frame for image list with scrollbar
        list_frame = ttk.Frame(scrollable_frame)
        list_frame.grid(row=5, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical")
        self.image_listbox = tk.Listbox(list_frame, height=6, yscrollcommand=list_scroll.set)
        list_scroll.config(command=self.image_listbox.yview)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Target droplets for selected image
        target_frame = ttk.Frame(scrollable_frame)
        target_frame.grid(row=6, column=0, columnspan=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(target_frame, text="Target for selected:").pack(side=tk.LEFT, padx=2)
        self.selected_target_var = tk.IntVar(value=3)
        ttk.Entry(target_frame, textvariable=self.selected_target_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_frame, text="Update", command=self.update_selected_target, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_frame, text="Set All", command=self.set_all_targets, width=8).pack(side=tk.LEFT, padx=2)
        
        # Bayesian Optimization Settings
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=7, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(scrollable_frame, text="Bayesian Optimization", font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=5, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Max Iterations:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.max_iterations_var = tk.IntVar(value=50)
        ttk.Entry(scrollable_frame, textvariable=self.max_iterations_var, width=10).grid(row=9, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(scrollable_frame, text="Error Metric:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.error_metric_var = tk.StringVar(value="average")
        error_combo = ttk.Combobox(scrollable_frame, textvariable=self.error_metric_var, 
                                    values=["average", "sum", "max"], width=10, state="readonly")
        error_combo.grid(row=10, column=1, sticky=tk.W, pady=5)
        
        opt_button_frame = ttk.Frame(scrollable_frame)
        opt_button_frame.grid(row=11, column=0, columnspan=5, pady=10)
        
        self.optimize_button = ttk.Button(opt_button_frame, text="🎯 Start Optimization", command=self.start_optimization, width=20)
        self.optimize_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_opt_button = ttk.Button(opt_button_frame, text="⏹ Stop", command=self.stop_optimization, width=15, state=tk.DISABLED)
        self.stop_opt_button.pack(side=tk.LEFT, padx=5)
        
        # Optimization progress
        self.opt_progress_var = tk.StringVar(value="Not started")
        ttk.Label(scrollable_frame, textvariable=self.opt_progress_var, foreground="blue", wraplength=380).grid(row=12, column=0, columnspan=5, sticky=tk.W)
        
        # ROI Settings
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=13, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(scrollable_frame, text="ROI Settings", font=('Arial', 10, 'bold')).grid(row=14, column=0, columnspan=5, sticky=tk.W)
        
        # Add header for lock column
        ttk.Label(scrollable_frame, text="Lock", font=('Arial', 8)).grid(row=15, column=4, sticky=tk.W)
        
        self.params = {}
        self.param_locks = {}
        self.param_bounds = {}
        row = 16
        
        # ROI parameters
        roi_params = [
            ("X_MIN", 1570, 0, 7000, 100),
            ("X_MAX", 5790, 0, 7000, 100),
            ("Y_MIN", 1980, 0, 5000, 100),
            ("Y_MAX", 3632, 0, 5000, 100),
        ]
        
        for param_name, default, min_val, max_val, step in roi_params:
            self.add_parameter_slider(scrollable_frame, row, param_name, default, min_val, max_val, step)
            row += 1
        
        # Detection Parameters
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Label(scrollable_frame, text="Detection Parameters", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=5, sticky=tk.W)
        row += 1
        
        # Add header for lock column
        ttk.Label(scrollable_frame, text="Lock", font=('Arial', 8)).grid(row=row, column=4, sticky=tk.W)
        row += 1
        
        detection_params = [
            ("BRIGHTNESS_PERCENTILE", 90.5, 80.0, 99.9, 0.5),
            ("MIN_BLOB_AREA", 80000, 10000, 500000, 5000),
            ("MAX_BLOB_AREA", 3000000, 100000, 5000000, 10000),
            ("BLUR_SIZE", 15, 3, 51, 2),  # Must be odd
            ("MORPH_KERNEL_SIZE", 15, 3, 51, 2),  # Must be odd
        ]
        
        for param_name, default, min_val, max_val, step in detection_params:
            self.add_parameter_slider(scrollable_frame, row, param_name, default, min_val, max_val, step)
            row += 1
        
        # Control buttons
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=5, pady=10)
        
        ttk.Button(button_frame, text="▶ Process Current", command=self.process_current_image, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="▶▶ Process All", command=self.process_all_images, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Reset", command=self.reset_to_defaults, width=12).pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        button_frame2 = ttk.Frame(scrollable_frame)
        button_frame2.grid(row=row, column=0, columnspan=5, pady=5)
        
        ttk.Button(button_frame2, text="📋 Copy Settings", command=self.copy_settings, width=15).pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        # Auto-update checkbox
        self.auto_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Auto-update on parameter change", 
                       variable=self.auto_update_var).grid(row=row, column=0, columnspan=5, sticky=tk.W)
        
        row += 1
        
        # Settings output
        ttk.Label(scrollable_frame, text="Python Code:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=5, sticky=tk.W, pady=(10,5))
        row += 1
        
        self.settings_text = scrolledtext.ScrolledText(scrollable_frame, height=8, width=50, wrap=tk.WORD)
        self.settings_text.grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        # Right panel - Image display
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image canvas
        self.canvas = tk.Canvas(right_frame, width=1200, height=800, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Load images to begin")
        status_label = ttk.Label(right_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def add_parameter_slider(self, parent, row, name, default, min_val, max_val, step=1):
        """Add a parameter with slider, entry, and lock checkbox"""
        ttk.Label(parent, text=f"{name}:", font=('Arial', 9)).grid(row=row, column=0, sticky=tk.W, pady=2, padx=(0,5))
        
        var = tk.DoubleVar(value=default)
        self.params[name] = var
        self.param_bounds[name] = (min_val, max_val, step)
        
        # Entry box
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky=tk.W, padx=3, pady=2)
        
        # Slider
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, 
                          orient=tk.HORIZONTAL, command=lambda v: self.on_param_change(name))
        slider.grid(row=row, column=2, sticky=(tk.W, tk.E), padx=3, pady=2)
        
        # Range label - REDUCED FONT SIZE
        ttk.Label(parent, text=f"[{min_val}-{max_val}]", font=('Arial', 6)).grid(row=row, column=3, sticky=tk.W, pady=2, padx=(0,5))
        
        # Lock checkbox - MORE PADDING
        lock_var = tk.BooleanVar(value=True)  # Locked by default
        self.param_locks[name] = lock_var
        ttk.Checkbutton(parent, variable=lock_var).grid(row=row, column=4, pady=2, padx=(5,10))
        
        # Bind entry changes
        entry.bind('<Return>', lambda e: self.on_param_change(name))
    
    def on_param_change(self, param_name):
        """Called when a parameter changes"""
        # Ensure odd numbers for kernel sizes
        if 'SIZE' in param_name:
            val = int(self.params[param_name].get())
            if val % 2 == 0:
                val += 1
            self.params[param_name].set(val)
        
        if self.auto_update_var.get() and len(self.loaded_images) > 0 and not self.optimizing:
            self.process_current_image()
    
    def browse_script(self):
        filename = filedialog.askopenfilename(
            title="Select detection script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename:
            self.script_path_var.set(filename)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_folder_var.set(folder)
    
    def load_images(self):
        """Load all images from input folder"""
        input_folder = self.input_folder_var.get()
        
        if not os.path.exists(input_folder):
            self.status_var.set(f"❌ Folder not found: {input_folder}")
            return
        
        # Find images
        patterns = [
            os.path.join(input_folder, "*.jpg"),
            os.path.join(input_folder, "*.JPG"),
            os.path.join(input_folder, "*.png"),
            os.path.join(input_folder, "*.PNG")
        ]
        
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(pattern))
        
        # FIXED: Remove duplicates (case-insensitive file systems match both *.jpg and *.JPG)
        all_files = list(set(all_files))
        
        if not all_files:
            self.status_var.set(f"❌ No images found in {input_folder}")
            return
        
        # Load all images
        self.loaded_images = []
        for image_path in sorted(all_files):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                filename = os.path.basename(image_path)
                self.loaded_images.append({
                    'filename': filename,
                    'image': img,
                    'target': 3,  # Default target
                    'last_detected': None
                })
        
        if len(self.loaded_images) == 0:
            self.status_var.set(f"❌ Could not load any images")
            return
        
        # Update listbox
        self.image_listbox.delete(0, tk.END)
        for img_data in self.loaded_images:
            self.image_listbox.insert(tk.END, f"{img_data['filename']} (target: {img_data['target']})")
        
        # Select first image
        self.current_image_index = 0
        self.image_listbox.selection_set(0)
        self.update_image_counter()
        self.display_current_image()
        
        self.status_var.set(f"✓ Loaded {len(self.loaded_images)} images")
    
    def on_image_select(self, event):
        """Called when an image is selected in the listbox"""
        selection = self.image_listbox.curselection()
        if selection:
            self.current_image_index = selection[0]
            self.selected_target_var.set(self.loaded_images[self.current_image_index]['target'])
            self.display_current_image()
    
    def update_selected_target(self):
        """Update target for currently selected image"""
        if len(self.loaded_images) == 0:
            return
        
        new_target = self.selected_target_var.get()
        self.loaded_images[self.current_image_index]['target'] = new_target
        
        # Update listbox
        img_data = self.loaded_images[self.current_image_index]
        self.image_listbox.delete(self.current_image_index)
        self.image_listbox.insert(self.current_image_index, f"{img_data['filename']} (target: {img_data['target']})")
        self.image_listbox.selection_set(self.current_image_index)
    
    def set_all_targets(self):
        """Set all images to the same target"""
        if len(self.loaded_images) == 0:
            return
        
        target = self.selected_target_var.get()
        
        for i, img_data in enumerate(self.loaded_images):
            img_data['target'] = target
            self.image_listbox.delete(i)
            self.image_listbox.insert(i, f"{img_data['filename']} (target: {target})")
        
        self.image_listbox.selection_set(self.current_image_index)
    
    def prev_image(self):
        """Go to previous image"""
        if len(self.loaded_images) == 0:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.loaded_images)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_image_index)
        self.image_listbox.see(self.current_image_index)
        self.selected_target_var.set(self.loaded_images[self.current_image_index]['target'])
        self.display_current_image()
    
    def next_image(self):
        """Go to next image"""
        if len(self.loaded_images) == 0:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.loaded_images)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_image_index)
        self.image_listbox.see(self.current_image_index)
        self.selected_target_var.set(self.loaded_images[self.current_image_index]['target'])
        self.display_current_image()
    
    def update_image_counter(self):
        """Update the image counter display"""
        if len(self.loaded_images) == 0:
            self.image_counter_var.set("No images loaded")
        else:
            self.image_counter_var.set(f"Image {self.current_image_index + 1} of {len(self.loaded_images)}")
    
    def display_current_image(self):
        """Display the current image"""
        if len(self.loaded_images) == 0:
            return
        
        img_data = self.loaded_images[self.current_image_index]
        self.display_image(img_data['image'], f"{img_data['filename']}")
        self.update_image_counter()
    
    def process_current_image(self):
        """Process only the currently displayed image"""
        if len(self.loaded_images) == 0:
            self.status_var.set("❌ No images loaded")
            return
        
        img_data = self.loaded_images[self.current_image_index]
        detected = self.process_single_image(img_data['image'], img_data['filename'])
        img_data['last_detected'] = detected
        
        target = img_data['target']
        error = abs(detected - target)
        
        self.status_var.set(f"✓ {img_data['filename']}: Detected {detected}, Target {target}, Error {error}")
    
    def process_all_images(self):
        """Process all loaded images and show results"""
        if len(self.loaded_images) == 0:
            self.status_var.set("❌ No images loaded")
            return
        
        self.status_var.set("⏳ Processing all images...")
        
        results = []
        for img_data in self.loaded_images:
            detected = self.process_single_image(img_data['image'], img_data['filename'], visualize=False)
            img_data['last_detected'] = detected
            target = img_data['target']
            error = abs(detected - target)
            results.append((img_data['filename'], detected, target, error))
        
        # Display summary
        total_error = sum(r[3] for r in results)
        avg_error = total_error / len(results)
        
        summary = f"✓ Processed {len(results)} images. Avg error: {avg_error:.2f}\n"
        for filename, detected, target, error in results:
            summary += f"  {filename}: {detected}/{target} (err: {error})\n"
        
        self.status_var.set(summary[:200])  # Truncate if too long
        
        # Update listbox with results
        self.image_listbox.delete(0, tk.END)
        for img_data in self.loaded_images:
            detected = img_data.get('last_detected', '?')
            self.image_listbox.insert(tk.END, f"{img_data['filename']} (target: {img_data['target']}, detected: {detected})")
        
        # Re-select current
        self.image_listbox.selection_set(self.current_image_index)
        
        # Display current image with detections
        self.process_current_image()
    
    def process_single_image(self, image, filename, params_dict=None, visualize=True):
        """Process a single image and return droplet count"""
        # Get parameters (either from provided dict or from sliders)
        if params_dict is None:
            x_min = int(self.params['X_MIN'].get())
            x_max = int(self.params['X_MAX'].get())
            y_min = int(self.params['Y_MIN'].get())
            y_max = int(self.params['Y_MAX'].get())
            brightness_percentile = self.params['BRIGHTNESS_PERCENTILE'].get()
            min_area = int(self.params['MIN_BLOB_AREA'].get())
            max_area = int(self.params['MAX_BLOB_AREA'].get())
            blur_size = int(self.params['BLUR_SIZE'].get())
            morph_size = int(self.params['MORPH_KERNEL_SIZE'].get())
        else:
            x_min = int(params_dict.get('X_MIN', self.params['X_MIN'].get()))
            x_max = int(params_dict.get('X_MAX', self.params['X_MAX'].get()))
            y_min = int(params_dict.get('Y_MIN', self.params['Y_MIN'].get()))
            y_max = int(params_dict.get('Y_MAX', self.params['Y_MAX'].get()))
            brightness_percentile = params_dict.get('BRIGHTNESS_PERCENTILE', self.params['BRIGHTNESS_PERCENTILE'].get())
            min_area = int(params_dict.get('MIN_BLOB_AREA', self.params['MIN_BLOB_AREA'].get()))
            max_area = int(params_dict.get('MAX_BLOB_AREA', self.params['MAX_BLOB_AREA'].get()))
            blur_size = int(params_dict.get('BLUR_SIZE', self.params['BLUR_SIZE'].get()))
            morph_size = int(params_dict.get('MORPH_KERNEL_SIZE', self.params['MORPH_KERNEL_SIZE'].get()))
        
        try:
            # Ensure odd sizes
            if blur_size % 2 == 0:
                blur_size += 1
            if morph_size % 2 == 0:
                morph_size += 1
            
            # Crop ROI
            roi = image[y_min:y_max, x_min:x_max]
            
            # Blur
            blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
            
            # Threshold
            threshold_value = np.percentile(blurred, brightness_percentile)
            _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Count valid droplets
            droplet_count = 0
            valid_droplets = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    cx, cy = centroids[i]
                    cx, cy = int(cx) + x_min, int(cy) + y_min
                    radius = int(np.sqrt(area / np.pi))
                    valid_droplets.append((cx, cy, radius, area))
                    droplet_count += 1
            
            # Only update visualization if requested and not optimizing
            if visualize and not self.optimizing:
                # Create visualization
                result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
                
                # Draw ROI
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # Draw detections
                for idx, (cx, cy, radius, area) in enumerate(valid_droplets, 1):
                    cv2.circle(result, (cx, cy), radius, (0, 0, 255), 4)
                    cv2.circle(result, (cx, cy), 8, (255, 0, 0), -1)
                    label = f"#{idx}"
                    cv2.putText(result, label, (cx + radius + 10, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Add summary text
                text = f"DROPLETS DETECTED: {droplet_count}"
                color = (0, 0, 255) if droplet_count > 0 else (0, 255, 0)
                cv2.putText(result, text, (x_min, y_min - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                
                # Display result
                self.display_image(result, f"{filename} (Threshold: {threshold_value:.0f})")
            
            return droplet_count
            
        except Exception as e:
            if not self.optimizing:
                print(f"Error processing {filename}: {str(e)}")
            return 0
    
    def start_optimization(self):
        """Start Bayesian optimization across all images"""
        if len(self.loaded_images) == 0:
            messagebox.showerror("Error", "Please load images first!")
            return
        
        if self.optimizing:
            return
        
        # Check that all images have valid targets
        for img_data in self.loaded_images:
            if img_data['target'] < 0:  # CHANGED: Now allows 0, only rejects negative
                messagebox.showerror("Error", "All target droplet counts must be >= 0")
                return
        
        # Get unlocked parameters
        unlocked_params = []
        for param_name, lock_var in self.param_locks.items():
            if not lock_var.get():  # If not locked
                unlocked_params.append(param_name)
        
        if len(unlocked_params) == 0:
            messagebox.showerror("Error", "Please unlock at least one parameter to optimize!")
            return
        
        self.optimizing = True
        self.optimize_button.config(state=tk.DISABLED)
        self.stop_opt_button.config(state=tk.NORMAL)
        
        # Run optimization in thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization,
            args=(unlocked_params,)
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
    
    def stop_optimization(self):
        """Stop ongoing optimization"""
        self.optimizing = False
        self.opt_progress_var.set("Stopping optimization...")
    
    def _run_optimization(self, unlocked_params):
        """Run Bayesian optimization in background thread"""
        try:
            # Build search space for unlocked parameters only
            space = []
            param_names = []
            
            for param_name in unlocked_params:
                min_val, max_val, step = self.param_bounds[param_name]
                param_names.append(param_name)
                
                # Use Integer space for integer parameters
                if param_name in ['X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'MIN_BLOB_AREA', 'MAX_BLOB_AREA', 'BLUR_SIZE', 'MORPH_KERNEL_SIZE']:
                    space.append(Integer(int(min_val), int(max_val), name=param_name))
                else:
                    space.append(Real(min_val, max_val, name=param_name))
            
            # Objective function
            iteration_count = [0]
            best_params = [None]
            best_score = [float('inf')]
            
            error_metric = self.error_metric_var.get()
            
            def objective(params):
                if not self.optimizing:
                    return best_score[0]  # Return last best score to stop gracefully
                
                iteration_count[0] += 1
                
                # Build full parameter dict
                params_dict = {}
                for i, param_name in enumerate(param_names):
                    value = params[i]
                    # Ensure odd for size parameters
                    if 'SIZE' in param_name and int(value) % 2 == 0:
                        value = int(value) + 1
                    params_dict[param_name] = value
                
                # Add locked parameters
                for param_name, lock_var in self.param_locks.items():
                    if lock_var.get():  # If locked
                        params_dict[param_name] = self.params[param_name].get()
                
                # Run detection on ALL images
                errors = []
                results_summary = []
                for img_data in self.loaded_images:
                    detected = self.process_single_image(img_data['image'], img_data['filename'], params_dict, visualize=False)
                    target = img_data['target']
                    error = abs(detected - target)
                    errors.append(error)
                    results_summary.append(f"{img_data['filename'][:15]}:{detected}/{target}")
                
                # Calculate aggregate error based on metric
                if error_metric == "average":
                    total_error = np.mean(errors)
                elif error_metric == "sum":
                    total_error = np.sum(errors)
                else:  # max
                    total_error = np.max(errors)
                
                # Update progress
                progress_text = f"Iter {iteration_count[0]}/{self.max_iterations_var.get()}: {error_metric} error={total_error:.2f} | {', '.join(results_summary[:3])}"
                
                if total_error < best_score[0]:
                    best_score[0] = total_error
                    best_params[0] = params_dict.copy()
                    progress_text += " ⭐ BEST!"
                
                self.opt_progress_var.set(progress_text)
                
                return total_error
            
            # Run optimization
            max_iter = self.max_iterations_var.get()
            self.opt_progress_var.set(f"Starting optimization for {len(unlocked_params)} params across {len(self.loaded_images)} images...")
            
            result = gp_minimize(
                objective,
                space,
                n_calls=max_iter,
                random_state=42,
                verbose=False
            )
            
            # Update parameters with best found
            if best_params[0] is not None and self.optimizing:
                for param_name, value in best_params[0].items():
                    if param_name in self.params:
                        self.params[param_name].set(value)
                
                # Process all images to show final result
                self.process_all_images()
                
                self.opt_progress_var.set(
                    f"✓ Optimization complete! Best {error_metric} error: {best_score[0]:.2f}"
                )
            else:
                self.opt_progress_var.set("Optimization stopped by user")
            
        except Exception as e:
            self.opt_progress_var.set(f"❌ Optimization error: {str(e)}")
        finally:
            self.optimizing = False
            self.optimize_button.config(state=tk.NORMAL)
            self.stop_opt_button.config(state=tk.DISABLED)
    
    def display_image(self, cv_image, title=""):
        """Display OpenCV image on canvas"""
        # Convert to RGB
        if len(cv_image.shape) == 2:
            display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet rendered
            canvas_width = 1200
            canvas_height = 800
        
        h, w = display_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.95
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(resized)
        self.current_display_image = ImageTk.PhotoImage(image=pil_image)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                anchor=tk.CENTER, image=self.current_display_image)
    
    def reset_to_defaults(self):
        """Reset all parameters to defaults"""
        defaults = {
            'X_MIN': 1570,
            'X_MAX': 5790,
            'Y_MIN': 1980,
            'Y_MAX': 3632,
            'BRIGHTNESS_PERCENTILE': 96.66080167002514,
            'MIN_BLOB_AREA': 94603,
            'MAX_BLOB_AREA': 4598289,
            'BLUR_SIZE': 51,
            'MORPH_KERNEL_SIZE': 37,
        }
        
        for param_name, value in defaults.items():
            if param_name in self.params:
                self.params[param_name].set(value)
        
        self.status_var.set("✓ Reset to default parameters")
    
    def copy_settings(self):
        """Generate Python code for current settings and copy to clipboard"""
        code = "# Detection Parameters\n"
        code += f"X_MIN, X_MAX = {int(self.params['X_MIN'].get())}, {int(self.params['X_MAX'].get())}\n"
        code += f"Y_MIN, Y_MAX = {int(self.params['Y_MIN'].get())}, {int(self.params['Y_MAX'].get())}\n"
        code += f"BRIGHTNESS_PERCENTILE = {self.params['BRIGHTNESS_PERCENTILE'].get()}\n"
        code += f"MIN_BLOB_AREA = {int(self.params['MIN_BLOB_AREA'].get())}\n"
        code += f"MAX_BLOB_AREA = {int(self.params['MAX_BLOB_AREA'].get())}\n"
        code += f"BLUR_SIZE = {int(self.params['BLUR_SIZE'].get())}\n"
        code += f"MORPH_KERNEL_SIZE = {int(self.params['MORPH_KERNEL_SIZE'].get())}\n"
        
        # Display in text box
        self.settings_text.delete(1.0, tk.END)
        self.settings_text.insert(1.0, code)
        
        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        
        self.status_var.set("✓ Settings copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DropletDetectorTuner(root)
    root.mainloop()