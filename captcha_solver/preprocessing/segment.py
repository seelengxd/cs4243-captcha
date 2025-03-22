from collections import Counter
import os
import random
import cv2
import multiprocessing
import concurrent.futures
import string
import sys
import numpy as np

sys.setrecursionlimit(10**6)





def test_image(img):
    img = img.copy()
    # print(img.shape)

    # Flatten image for faster counting
    flat_pixels = img.reshape(-1, 3)

    # Count unique colors
    unique, counts = np.unique(flat_pixels, axis=0, return_counts=True)
    max_colors = 10

    # Get most common colors (excluding black [0, 0, 0])
    most_common_colors = unique[np.argsort(-counts)][:max_colors + 2]
    most_common_colors = [color for color in most_common_colors if not np.array_equal(color, [0, 0, 0])]

    # Identify black pixels
    black_mask = (img == [0, 0, 0]).all(axis=-1)

    # Process black pixels
    if np.any(black_mask):
        padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)

        # Iterate over black pixels only
        black_indices = np.argwhere(black_mask)

        for x, y in black_indices:
            # Extract 3x3 neighborhood
            neighborhood = padded_img[x:x + 3, y:y + 3].reshape(-1, 3)

            # Filter neighborhood with most common colors
            valid_neighbors = [tuple(pixel) for pixel in neighborhood if tuple(pixel) in map(tuple, most_common_colors)]

            if valid_neighbors:
                # Find most common color in the neighborhood
                fill_pixel = Counter(valid_neighbors).most_common(1)[0][0]
                img[x, y] = fill_pixel
    return img

def color_letters_black(img):
    img = img.copy()

    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if not (pixel == [255, 255, 255]).all():
                img[i][j] = [0, 0, 0]
            else:
                img[i][j] = [255, 255, 255]
    return img


def merge_boxes(box1, box2):
    return [min(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), max(box1[3], box2[3]), box1[4]]

def is_surrounded_horizontally(box1, box2, atol=10)-> bool:
    """return box1 horizontally is inside box2"""
    return box1[2] >= box2[2] - atol and box1[3] <= box2[3] + atol

def is_tiny_box(box):
    return (abs(box[1] - box[0]) * abs(box[3] - box[2])) < 80

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

def partition_box(box, n=2):
    # print(type(box[3]))
    # print(box[3], box[2], n, (box[3] - box[2]) // n)
    part_length = int((box[3] - box[2]) // n)
    # print(part_length)
    return [[box[0], box[1], box[2] + i * part_length, box[2] + (i + 1) * part_length] for i in range(n)]

def pixels_are_close(pixel1, pixel2, atol=20):
    """
    Determines if two RGB pixels are close in color, allowing for slight variations.
    
    Uses Euclidean distance in the RGB color space.
    
    Args:
        pixel1 (array-like): First pixel (e.g., [R, G, B]).
        pixel2 (array-like): Second pixel (e.g., [R, G, B]).
        atol (int): Acceptable color difference threshold.
        
    Returns:
        bool: True if the pixels are similar, False otherwise.
    """
    return np.linalg.norm(np.array(pixel1) - np.array(pixel2)) < atol

def merge_all_boxes(bounding_boxes, close_threshold=30, debug=False):
    merged_boxes = []
    for box in bounding_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            if is_surrounded_horizontally(box, merged_boxes[-1]) or is_surrounded_horizontally(merged_boxes[-1], box):
                # print("surrounded", abs(box[1] - box[0]) * abs(box[3] - box[2]))
                merged_boxes[-1] = merge_boxes(merged_boxes[-1], box)
                # print("tiny", abs(box[1] - box[0]) * abs(box[3] - box[2]))
                # merged_boxes[-1] = merge_boxes(merged_boxes[-1], box)
                # Note: we don't append tiny boxes for now, seem to be causing the effect
                # pass
            elif pixels_are_close(box[4], merged_boxes[-1][4]) and box[2] - merged_boxes[-1][3] < close_threshold:
                if debug:
                    print("close", box[4], merged_boxes[-1][4])
                merged_boxes[-1] = merge_boxes(merged_boxes[-1], box)
            elif (is_tiny_box(box) or is_tiny_box(merged_boxes[-1])) and box[2] - merged_boxes[-1][3] < close_threshold:
                pass
            else:
                if debug:
                    print("not close", box[4], merged_boxes[-1][4], np.sum(np.abs(box[4] - merged_boxes[-1][4])))
                # if pixels_are_close(box[4], merged_boxes[-1][4]):
                    # print(box, merged_boxes[-1], box[2] - merged_boxes[-1][3], "whats going on")
               
                # print("not tiny", abs(box[1] - box[0]) * abs(box[3] - box[2]))
                merged_boxes.append(box)
    return merged_boxes

def get_bounding_boxes(img):
    # Assume this is the cleaned black letter image

    img = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype("int32")

    shape = img.shape[:2]
    visited = np.zeros(shape, dtype=bool)
    bounding_boxes = [] # top bottom left right color

    def dfs(i, j, color):
        if i < 0 or j < 0 or i >= shape[0] or j >= shape[1] or visited[i][j]:
            return
        if pixels_are_close(img[i][j], [255, 255, 255]):
            return
        # if color is not close enough return
        if not pixels_are_close(img[i][j], color):
            return  
        visited[i][j] = True
        bounding_boxes[-1][0] = min(bounding_boxes[-1][0], i)
        bounding_boxes[-1][1] = max(bounding_boxes[-1][1], i)
        bounding_boxes[-1][2] = min(bounding_boxes[-1][2], j)
        bounding_boxes[-1][3] = max(bounding_boxes[-1][3], j)

        # color = img[i][j]
        dfs(i + 1, j + 1, color)
        dfs(i + 1, j, color)
        dfs(i + 1, j - 1, color)

        dfs(i - 1, j - 1, color)
        dfs(i - 1, j, color)
        dfs(i - 1, j + 1, color)
       
        dfs(i, j + 1, color)
        dfs(i, j - 1, color)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if visited[i][j] or pixels_are_close(img[i][j], [255, 255, 255]):
                continue
            bounding_boxes.append([i, i, j, j, img[i][j]])
            dfs(i, j, img[i][j])

    bounding_boxes = [box for box in bounding_boxes if box[0] != box[1] and box[2] != box[3]]
    
    # Merge boxes
    bounding_boxes.sort(key=lambda x: x[2])
    close_threshold = 8 #0.025 * shape[1]
    # print("close threshold", int(close_threshold))
    merged_boxes = merge_all_boxes(merge_all_boxes(bounding_boxes, close_threshold=close_threshold), close_threshold=close_threshold)
    

    # Split boxes
    horizontal_lengths = [abs(box[3] - box[2]) for box in merged_boxes]
    
    # print(len(merged_boxes))
    for box in merged_boxes:
        # print(box)
        cv2.rectangle(img, (box[2], box[0]), (box[3], box[1]), [255, 0, 0], 1)
        # print(box)

    return img, merged_boxes

# random_image_name = "rqextpir-0.png" or fetch_random_image()
# failures
# mkwuh4-0.png
# random_image_name = fetch_random_image()
# print(random_image_name)
# random_image =  read_image(random_image_name)
# cleaned_lines = test_image(random_image)
# black_letters = color_letters_black(cleaned_lines)
# boxes, box_data = get_bounding_boxes(cleaned_lines)


debug = open("debug.txt", "w")
print("hello??", file=debug, flush=True)
def process_image(image, input_folder, output_folder, lock, doesntfit):
    if not image.endswith(".png"):
        return
    image_path = os.path.join(input_folder, image)
    cleaned = test_image(cv2.imread(image_path))  
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


def extract_characters(input_folder, output_folder):
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

    print(list(doesntfit))  # Convert `Manager().list()` to a normal list before printing


# Run the function
if __name__ == "__main__":
    multiprocessing.freeze_support()
    extract_characters("../data/train_cleaned_color_resized", "../data/train_cleaned_characters")
    extract_characters("../data/test_cleaned_color_resized", "../data/test_cleaned_characters")
