import os
import zipfile
import glob

def restore_kg():
    # Define paths
    base_dir = r'C:\bitirme3\ENS491-MACKIS\creating_kg\knowledge_graph'
    zip_path = os.path.join(base_dir, 'kg_llm_validated.zip')
    extract_dir = os.path.join(base_dir, 'llm_validated')
    
    # 1. Find all parts
    parts = sorted(glob.glob(f"{zip_path}.part*"))
    
    if not parts:
        print(f"Error: No zip parts found matching {zip_path}.part*")
        return
        
    print(f"Found {len(parts)} parts. Merging...")
    
    # 2. Merge parts into the original large zip
    with open(zip_path, 'wb') as outfile:
        for part in parts:
            print(f"  Reading {os.path.basename(part)}...")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
                
    print("\nMerge complete. Extracting files...")
    
    # 3. Extract the merged zip
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for item in zf.namelist():
            print(f"  Extracting {item}...")
        zf.extractall(extract_dir)
        
    print(f"Extraction complete to {extract_dir}.\n")
    
    # 4. Clean up the temporary merged zip file
    os.remove(zip_path)
    print("Cleaned up the temporary merged zip file.")
    print("Knowledge Graph restored successfully! You can now use your KG locally.")

if __name__ == '__main__':
    restore_kg()
