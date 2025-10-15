import os
import sys
from send2trash import send2trash
import shutil

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


def get_trash_path():
    """Return the system trash directory depending on OS."""
    if sys.platform.startswith("win"):
        # Windows recycle bin root (will contain subfolders for each user SID)
        return os.path.expandvars(r"%SYSTEMDRIVE%\$Recycle.Bin")
    elif sys.platform == "darwin":
        # macOS Trash
        return os.path.expanduser("~/.Trash")
    else:
        # Linux freedesktop Trash spec
        return os.path.expanduser("~/.local/share/Trash/files")


def count_items_in_trash():
    """Count how many items are currently in the system trash folder."""
    trash_path = get_trash_path()
    try:
        if not os.path.exists(trash_path):
            return 0
        # On Windows, $Recycle.Bin contains subfolders; count recursively
        total = 0
        for root, dirs, files in os.walk(trash_path):
            total += len(files) + len(dirs)
        return total
    except Exception as e:
        print(f"⚠️ Could not count trash contents: {e}")
        return 0


def clear_folder(folder_path):
    """Move all files in a folder to the system trash."""
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

    # ✅ Check current trash contents after clearing
    total_in_trash = count_items_in_trash()
    print(f"\n🗑️ Current items in system trash: {total_in_trash}")

    if total_in_trash > 100:
        choice = input("⚠️ OVER 100 ITEMS IN TRASH — EMPTY NOW? (Y/N): ").strip().lower()
        if choice == "y":
            print("🧹 Emptying system trash...")
            if sys.platform.startswith("win"):
                try:
                    import winshell
                    winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=False)
                    print("✅ Recycle Bin emptied.")
                except Exception as e:
                    print(f"❌ Could not empty Recycle Bin: {e}")
            elif sys.platform == "darwin":
                os.system('osascript -e "tell application \\"Finder\\" to empty trash"')
                print("✅ macOS Trash emptied.")
            else:
                trash_path = get_trash_path()
                shutil.rmtree(trash_path, ignore_errors=True)
                os.makedirs(trash_path, exist_ok=True)
                print("✅ Linux Trash emptied.")
        else:
            print("🕳️ Trash not emptied.")
    else:
        print("✅ Trash level under limit, no action needed.")

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
