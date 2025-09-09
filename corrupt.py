from PIL import Image
import os
root = r"data\train"
corrupted = []
for r, d, files in os.walk(root):
    for f in files:
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            p = os.path.join(r,f)
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as e:
                corrupted.append(p)
                print(f"Corrupted: {p} - {e}")
print(f"Total corrupted: {len(corrupted)}")