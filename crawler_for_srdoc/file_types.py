# list_file_types.py
import os
import collections

# TODO: change this to your actual crawl root if needed
ROOT_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\crawler_for_srdoc\\mysu_dump_plus2"

def main():
    ext_counter = collections.Counter()
    total_files = 0

    for root, dirs, files in os.walk(ROOT_DIR):
        for fn in files:
            total_files += 1
            _, ext = os.path.splitext(fn)
            ext = ext.lower()
            if not ext:
                ext = "<no_ext>"
            ext_counter[ext] += 1

    print(f"Scanned {total_files} files under: {ROOT_DIR}\n")
    print("Unique extensions (sorted by count desc):")
    for ext, cnt in ext_counter.most_common():
        print(f"{ext:10s}  {cnt:5d}")

if __name__ == "__main__":
    main()
