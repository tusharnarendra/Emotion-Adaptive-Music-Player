import os
import json
import csv
import sys

dataset_root = 'AFEW-VA'  # Replace with your root folder containing the 600 folders
output_csv = 'VA_scores.csv'
failed_detections_file = 'failed_detections.txt'
max_images = 5000  # Limit to first 5000 images attempted

# Load failed detections into a set
with open(failed_detections_file, 'r') as f:
    failed_detections = set(line.strip() for line in f if line.strip())

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'valence', 'arousal'])

    count = 0  # Counter for total images attempted (written or skipped)

    for i in range(1, 601):
        folder_name = f"{i:03d}"
        json_path = os.path.join(dataset_root, folder_name, f"{folder_name}.json")
        
        if not os.path.isfile(json_path):
            print(f"Skipping missing file: {json_path}")
            sys.exit()
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        frames = data.get('frames', {})
        for frame_num, frame_data in frames.items():
            if count >= max_images:
                print(f"Reached limit of {max_images} images.")
                sys.exit()

            valence = frame_data.get('valence')
            arousal = frame_data.get('arousal')
            image_name = f"{folder_name}_{frame_num}.png"  # Image name with folder prefix

            # Skip if this image failed detection
            if image_name in failed_detections:
                count += 1
                continue

            writer.writerow([image_name, valence, arousal])
            count += 1

print(f"Finished processing {count} images (written + skipped).")
