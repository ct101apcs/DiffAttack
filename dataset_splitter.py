# The dataset is labeled 1.png, 2.png, ..., 1000.png
# Split into 2 equal parts for parallel processing
import os
import shutil
def split_dataset(input_dir, output_dir1, output_dir2, total_files=662):
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)

    for i in range(1, total_files + 1):
        filename = f"{i}.png"
        src_path = os.path.join(input_dir, filename)
        if i <= total_files // 2:
            dest_path = os.path.join(output_dir1, filename)
        else:
            dest_path = os.path.join(output_dir2, filename)
        
        shutil.copy(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")
        
# Example usage
input_directory = "datasets/imagenet-compatible/images"
output_directory1 = "datasets/imagenet-compatible/tmp/images_part1"
output_directory2 = "datasets/imagenet-compatible/tmp/images_part2"

# Create the directories if they don't exist
os.makedirs(output_directory1, exist_ok=True)
os.makedirs(output_directory2, exist_ok=True)

split_dataset(input_directory, output_directory1, output_directory2)