
import multiprocessing
import os
import string
import cv2
import concurrent.futures
import numpy as np

INPUT_FOLDERS = ["../data/train_cleaned_characters", "../data/test_cleaned_characters"]
OUTPUT_FOLDERS = ["../data/train_cleaned_characters_resized", "../data/test_cleaned_characters_resized"]

def get_image_size(image):
    return cv2.imread(image).shape

def extract_sizes(image, sizes, lock):
    image_size = get_image_size(image)
    with lock:
        sizes.append(image_size)

# 2. Resize all images to the same size. Maintain aspect ratio of the original image. Save to folder.
def resize_with_padding(image, target_width, target_height):
    """
    Resizes an image while maintaining aspect ratio, 
    scaling it to fit within target dimensions, and padding the rest with white.

    Parameters:
        image (numpy.ndarray): Input image.
        target_width (int): Target width.
        target_height (int): Target height.

    Returns:
        numpy.ndarray: Resized and padded image.
    """
    h, w = image.shape[:2]
    
    # Compute scaling factor to fit within the target size
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a white canvas
    result = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # Compute top-left corner for centering
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # Place resized image onto the white canvas
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result

def resize(input_folder, output_folder, character_folder, image, target_width, target_height):
    image_path = os.path.join(input_folder, character_folder, image)
    output_path = os.path.join(output_folder, character_folder, image)
    img = cv2.imread(image_path)
    resized = resize_with_padding(img, target_width, target_height)
    os.makedirs(os.path.join(output_folder, character_folder), exist_ok=True)
    cv2.imwrite(output_path, resized)

def resize_folder(input_folder, output_folder, target_width, target_height):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for folder in os.listdir(input_folder):
            for image in os.listdir(os.path.join(input_folder, folder)):
                futures.append(executor.submit(resize, input_folder, output_folder, folder, image, target_width, target_height))

        from tqdm import tqdm
        total = len(futures)
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total, desc="Resizing images -" + input_folder):
            pass
    
    
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. Read all images and get the size.
    for input_folder, output_folder in zip(INPUT_FOLDERS, OUTPUT_FOLDERS):
        characters = string.ascii_lowercase + string.digits

        os.makedirs(output_folder, exist_ok=True)
        for character in characters:
            os.makedirs(os.path.join(output_folder, character), exist_ok=True)

        manager = multiprocessing.Manager()
        lock = manager.Lock()
        sizes = manager.list()

        image_files = []
        for folder in os.listdir(input_folder):
            image_files.extend([os.path.join(input_folder, folder, image) for image in os.listdir(os.path.join(input_folder, folder))])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(extract_sizes, image, sizes, lock)
                for image in image_files
            ]

            from tqdm import tqdm
            total = len(futures)
            for _ in tqdm(concurrent.futures.as_completed(futures), total=total, desc="Processing images"):
                pass

    # 169 x 78
    max_y = max([size[0] for size in sizes])
    max_x = max([size[1] for size in sizes])

    # 2. Resize all images to the same size. Maintain aspect ratio of the original image. Save to folder.
    resize_folder(INPUT_FOLDERS[0], OUTPUT_FOLDERS[0], max_x, max_y)
    resize_folder(INPUT_FOLDERS[1], OUTPUT_FOLDERS[1], max_x, max_y)


