import zipfile
import os
from pathlib import Path
def create_colab_zip():
    print("=" * 60)
    print("Creating ZIP file for Google Colab")
    print("=" * 60)
    project_root = Path(__file__).parent
    os.chdir(project_root)
    zip_filename = "Thesis_llama_colab.zip"
    include_items = [
        "pipeline/",
        "input/",
        "Tests/",
        "colab_stage_e.ipynb",
        "COLAB_Stage_C.ipynb",
        "requirements.txt",
        "README.md",
    ]
    exclude_folders = {
        "__pycache__",
        ".git",
        ".ipynb_checkpoints",
        "llm",
        "src",
        "papers"
    }
    exclude_extensions = {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll"
    }
    exclude_files = set()
    print(f"\n[*] Creating: {zip_filename}")
    print(f"[*] Working directory: {os.getcwd()}\n")
    missing_items = []
    for item in include_items:
        if not os.path.exists(item):
            missing_items.append(item)
    if missing_items:
        print("[!] Warning: Some items not found:")
        for item in missing_items:
            print(f"   - {item}")
        print()
    total_files = 0
    total_size = 0
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_items:
            if not os.path.exists(item):
                continue
            if os.path.isfile(item):
                zipf.write(item)
                size = os.path.getsize(item)
                total_size += size
                total_files += 1
                print(f"[+] Added: {item} ({size / 1024:.1f} KB)")
            elif os.path.isdir(item):
                print(f"\n[*] Adding folder: {item}")
                folder_files = 0
                folder_size = 0
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in exclude_folders]
                    for file in files:
                        if any(file.endswith(ext) for ext in exclude_extensions):
                            continue
                        if file in exclude_files:
                            continue
                        filepath = os.path.join(root, file)
                        arcname = filepath
                        zipf.write(filepath, arcname)
                        size = os.path.getsize(filepath)
                        total_size += size
                        folder_size += size
                        total_files += 1
                        folder_files += 1
                print(f"    [+] {folder_files} files ({folder_size / (1024 * 1024):.2f} MB)")
    final_size = os.path.getsize(zip_filename)
    print("\n" + "=" * 60)
    print("[SUCCESS] ZIP file created successfully!")
    print("=" * 60)
    print(f"File: {zip_filename}")
    print(f"Total files: {total_files}")
    print(f"ZIP size: {final_size / (1024 * 1024):.2f} MB")
    print(f"Uncompressed: {total_size / (1024 * 1024):.2f} MB")
    print(f"Compression: {(1 - final_size / total_size) * 100:.1f}%")
    print("=" * 60)
    if final_size / (1024 * 1024) < 50:
        print("[OK] Good size for Colab upload!")
    elif final_size / (1024 * 1024) < 200:
        print("[!] Upload might be slow, but should work")
    else:
        print("[X] Too large for easy upload (>200MB)")
    print(f"\n[*] Ready to upload to Google Colab!")
    print(f"    Location: {os.path.join(os.getcwd(), zip_filename)}")
    return zip_filename
def list_zip_contents(zip_filename):
    print(f"\n\nZIP Contents Preview:")
    print("=" * 60)
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        folders = {}
        for name in zipf.namelist():
            top_folder = name.split('/')[0] if '/' in name else name
            if top_folder not in folders:
                folders[top_folder] = []
            folders[top_folder].append(name)
        for folder, files in sorted(folders.items()):
            print(f"\n[*] {folder}/")
            if len(files) <= 10:
                for f in files:
                    print(f"    - {f}")
            else:
                for f in files[:5]:
                    print(f"    - {f}")
                print(f"    ... and {len(files) - 5} more files")
if __name__ == "__main__":
    try:
        zip_filename = create_colab_zip()
        list_zip_contents(zip_filename)
        print("\n\n" + "=" * 60)
        print("[SUCCESS] Your ZIP is ready for Google Colab!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Go to Google Colab and upload this ZIP (Step 4 in the notebook)")
        print("2. Open colab_stage_e.ipynb from the extracted files (or upload it)")
        print("3. Runtime -> Change runtime type -> T4 GPU")
        print("4. Run all cells to execute Stage E (upload Stage D JSON, then run)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
