import os
import zipfile
import glob

def pack_and_split():
    # Define paths
    base_dir = r'C:\bitirme3\ENS491-MACKIS\creating_kg\knowledge_graph'
    llm_dir = os.path.join(base_dir, 'llm_validated')
    zip_path = os.path.join(base_dir, 'kg_llm_validated.zip')
    
    if not os.path.exists(llm_dir):
        print(f"Error: Directory {llm_dir} does not exist.")
        return

    # 1. Zip the files
    print(f"Creating zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for fname in os.listdir(llm_dir):
            fpath = os.path.join(llm_dir, fname)
            if os.path.isfile(fpath):
                print(f"  Compressing {fname}...")
                # We write it with just the filename, so it extracts directly into the target folder
                zf.write(fpath, fname)
            
    print("Zip created successfully.\n")
    
    # 2. Split the zip into 90 MB chunks (well below GitHub's 100MB limit)
    chunk_size = 90 * 1024 * 1024 # 90 MB
    part_num = 1
    
    # Clean up any old parts first
    old_parts = glob.glob(f"{zip_path}.part*")
    for old_part in old_parts:
        os.remove(old_part)
        
    print("Splitting the zip file into 90MB parts...")
    with open(zip_path, 'rb') as infile:
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            
            part_name = f"{zip_path}.part{part_num:03d}"
            with open(part_name, 'wb') as outfile:
                outfile.write(chunk)
            print(f"  Created {os.path.basename(part_name)} ({len(chunk)/1024/1024:.1f} MB)")
            part_num += 1
            
    # 3. Clean up the large zip file so it doesn't accidentally get pushed
    os.remove(zip_path)
    print(f"\nDone! The large zip was split into {part_num - 1} parts.")
    print("You can now safely commit and push the .part* files to GitHub.")

if __name__ == '__main__':
    pack_and_split()
