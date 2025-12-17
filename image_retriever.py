# The prefixes of images in folders are 0xxx, in which xxx is in range(001, 999)
# Given a list of folders, help me retrieve all image paths whose prefixes are in the given range, group by prefixes.
import os
from collections import defaultdict
from typing import List, Dict
def retrieve_image_paths(folders: List[str], start: int, end: int) -> Dict[str, List[str]]:
    image_paths = defaultdict(list)
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image file extensions
                prefix = filename.split('_')[0]  # Assuming the prefix is before the first underscore
                try:
                    prefix_num = int(prefix)
                    if start <= prefix_num <= end:
                        full_path = os.path.join(folder, filename)
                        image_paths[prefix].append(full_path)
                except ValueError:
                    continue  # Skip files that do not have a valid integer prefix
    return dict(image_paths)

def parse_folder_name(folder_name: str) -> str:
    # Extract model name from folder name
    parts = folder_name.split('_')
    if len(parts) >= 2:
        return parts[0]  # Return the model name part
    return folder_name

# Example usage:
folders = [ 'inception_demo_output_part1',
            'resnet_demo_output',
            'vgg_demo_output',
            'vit_demo_output',
            'swin_demo_output_part1']
output_path = 'retrieved_images'
os.makedirs(output_path, exist_ok=True)

result = retrieve_image_paths(folders, 1, 4)
for prefix, paths in result.items():
    print(f"Prefix: {prefix}")
    for path in paths:
        file_name = os.path.basename(path)
        model_name = parse_folder_name(os.path.dirname(path))
        new_path = os.path.join(output_path, prefix, model_name, file_name)
        # Copy the file to the output directory, grouped by prefix and 
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        with open(path, 'rb') as src_file:
            with open(new_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
                
        print(f"  {path} -> {new_path}")


