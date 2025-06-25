import os
import shutil
from pathlib import Path

source_root = Path('AFEW-VA') 
target_root = Path('dataset')
target_root.mkdir(exist_ok=True)

counter = 0
for folder in source_root.iterdir():
    if folder.is_dir():
        for img_path in folder.glob('*.png'):
            new_name = f"{folder.name}_{img_path.name}"
            shutil.copy(img_path, target_root / new_name)
            counter += 1

print(f"Copied {counter} images.")