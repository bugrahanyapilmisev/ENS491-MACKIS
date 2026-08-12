
import os
import zipfile
import shutil
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_V2 = os.getenv("CHROMA_DIR_V2", os.path.join(CURRENT_DIR, "chroma_db_v2"))
ZIP_PATH = os.path.join(CURRENT_DIR, "chroma_db_v2_compact.zip")


def main():
    print("=" * 60)
    print("RESTORE CHROMA DB V2 (from zip)")
    print("=" * 60)
    print(f"Zip Source : {ZIP_PATH}")
    print(f"Chroma Target: {CHROMA_DIR_V2}")
    print("=" * 60)

    # 1) Check zip exists
    if not os.path.exists(ZIP_PATH):
        print(f"[error] Zip file not found: {ZIP_PATH}")
        return

    # 2) Check if chroma_db_v2 already exists
    if os.path.exists(CHROMA_DIR_V2):
        existing_sqlite = os.path.join(CHROMA_DIR_V2, "chroma.sqlite3")
        if os.path.exists(existing_sqlite):
            size_mb = os.path.getsize(existing_sqlite) / (1024 * 1024)
            print(f"[warn] ChromaDB already exists ({size_mb:.1f} MB)")
            answer = input("Overwrite? (y/n): ").strip().lower()
            if answer != "y":
                print("Aborted.")
                return
            print("[info] Removing existing chroma_db_v2/ ...")
            shutil.rmtree(CHROMA_DIR_V2)

    # 3) Create target directory
    os.makedirs(CHROMA_DIR_V2, exist_ok=True)

    # 4) Extract zip
    print("[1/1] Extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(CHROMA_DIR_V2)

    # 5) Verify
    restored_sqlite = os.path.join(CHROMA_DIR_V2, "chroma.sqlite3")
    if os.path.exists(restored_sqlite):
        size_mb = os.path.getsize(restored_sqlite) / (1024 * 1024)
        print(f"\n✅ Restore finished! ({size_mb:.1f} MB)")
    else:
        print("\n❌ Restore failed — chroma.sqlite3 not found after extraction")



    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR_V2", os.path.join(CURRENT_DIR, "checkpoints_v2"))
    CHECKPOINT_ZIP_PATH = os.path.join(CURRENT_DIR, "checkpoint_files_zipped.zip")
    print("\n" + "=" * 60)
    print("RESTORE CHECKPOINT FILES (from zip)")
    print("=" * 60)
    print(f"Zip Source : {CHECKPOINT_ZIP_PATH}")
    print(f"Target Dir : {CHECKPOINT_DIR}")
    print("=" * 60)

    # 1) Check zip exists
    if not os.path.exists(CHECKPOINT_ZIP_PATH):
        print(f"[error] Zip file not found: {CHECKPOINT_ZIP_PATH}")
        return

    # 2) Check if checkpoints already exist
    if os.path.exists(CHECKPOINT_DIR):
        print(f"[warn] Checkpoint directory already exists: {CHECKPOINT_DIR}")
        answer = input("Overwrite? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return
        print(f"[info] Removing existing {CHECKPOINT_DIR} ...")
        shutil.rmtree(CHECKPOINT_DIR)

    # 3) Create target directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 4) Extract zip
    print("[1/1] Extracting zip...")
    with zipfile.ZipFile(CHECKPOINT_ZIP_PATH, "r") as zf:
        zf.extractall(CHECKPOINT_DIR)

if __name__ == "__main__":
    main()
