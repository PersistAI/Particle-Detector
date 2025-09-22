import os
from send2trash import send2trash

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDERS = [
    os.path.join(BASE_DIR, "image_process_input"),
    os.path.join(BASE_DIR, "image_process_output"),
    os.path.join(BASE_DIR, "image_process_archive"),
]
# ==============================

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping (folder not found): {folder_path}")
        return

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    if not files:
        print(f"✅ Nothing to clear in {folder_path}")
        return

    for f in files:
        try:
            send2trash(f)
            print(f"🗑️ Moved to trash: {f}")
        except Exception as e:
            print(f"❌ Could not move {f}: {e}")

def main():
    print("=== Clearing Particle Detector Folders ===")
    for folder in FOLDERS:
        clear_folder(folder)
    print("\n✅ All done.")

if __name__ == "__main__":
    main()
