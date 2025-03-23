"""
This script segments captchas into directories of individual letters:
data/test_cleaned_characters
data/train_cleaned_characters

```
cd captcha_solver
uv run -m preprocessing.segment
```
"""

import os
from preprocessing.utils import color_letters_black, get_bounding_boxes, remove_black_lines
import cv2
import multiprocessing
import concurrent.futures
import string
import sys
from multiprocessing.synchronize import Lock as LockBase

sys.setrecursionlimit(10**6)

def process_image(image: str, input_folder: str, output_folder: str, lock: LockBase, doesntfit: list[str]):
    """Given an image path, extract characters and save them to the output folder."""
    if not image.endswith(".png"):
        return
    image_path = os.path.join(input_folder, image)
    cleaned = remove_black_lines(cv2.imread(image_path))  
    _, boxes = get_bounding_boxes(cleaned) 
    label = image.split("-")[0]

    if len(boxes) != len(label):
        with lock:
            doesntfit.append(image)
        return

    for i, box in enumerate(boxes):
        x1, x2, y1, y2, _ = box
        cropped = cleaned[x1:x2, y1:y2]
        cropped = color_letters_black(cropped) 
        char_folder = os.path.join(output_folder, label[i])
        os.makedirs(char_folder, exist_ok=True)
        cv2.imwrite(os.path.join(char_folder, f"{label}-{i}.png"), cropped)


def extract_characters(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for char in string.ascii_letters + string.digits:
        os.makedirs(os.path.join(output_folder, char), exist_ok=True)

    # Use Manager once to create all shared resources
    manager = multiprocessing.Manager()
    doesntfit = manager.list()
    lock = manager.Lock()
    
    # Get list of image files first
    image_files = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    
    # Process images with progress tracking
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, image, input_folder, output_folder, lock, doesntfit)
            for image in image_files
        ]
        
        # Track progress
        from tqdm import tqdm
        total = len(futures)
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total, desc="Processing images"):
            pass

    print(*list(doesntfit), sep="\n")  

if __name__ == "__main__":
    multiprocessing.freeze_support()
    extract_characters("data/train_cleaned_color_resized", "data/train_cleaned_characters")
    extract_characters("data/test_cleaned_color_resized", "data/test_cleaned_characters")
