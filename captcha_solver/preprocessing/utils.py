# Helper functions
from collections import Counter
import random
import cv2
import numpy as np


def remove_black_lines(img):
    img = img.copy()

    # Flatten image for faster counting
    flat_pixels = img.reshape(-1, 3)

    # Count unique colors
    unique, counts = np.unique(flat_pixels, axis=0, return_counts=True)
    max_colors = 10

    # Get most common colors (excluding black [0, 0, 0])
    most_common_colors = unique[np.argsort(-counts)][: max_colors + 2]
    most_common_colors = [
        color for color in most_common_colors if not np.array_equal(color, [0, 0, 0])
    ]

    # Identify black pixels
    black_mask = (img == [0, 0, 0]).all(axis=-1)

    # Process black pixels
    if np.any(black_mask):
        padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)

        # Iterate over black pixels only
        black_indices = np.argwhere(black_mask)

        for x, y in black_indices:
            # Extract 3x3 neighborhood
            neighborhood = padded_img[x : x + 3, y : y + 3].reshape(-1, 3)

            # Filter neighborhood with most common colors
            valid_neighbors = [
                tuple(pixel)
                for pixel in neighborhood
                if tuple(pixel) in map(tuple, most_common_colors)
            ]

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
    return [
        min(box1[0], box2[0]),
        max(box1[1], box2[1]),
        min(box1[2], box2[2]),
        max(box1[3], box2[3]),
        box1[4],
    ]


def is_surrounded_horizontally(box1, box2, atol=10) -> bool:
    """return box1 horizontally is inside box2"""
    return box1[2] >= box2[2] - atol and box1[3] <= box2[3] + atol


def is_tiny_box(box):
    return (abs(box[1] - box[0]) * abs(box[3] - box[2])) < 80


def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def partition_box(box, n=2):
    part_length = int((box[3] - box[2]) // n)
    return [
        [box[0], box[1], box[2] + i * part_length, box[2] + (i + 1) * part_length]
        for i in range(n)
    ]


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
            if is_surrounded_horizontally(
                box, merged_boxes[-1]
            ) or is_surrounded_horizontally(merged_boxes[-1], box):
                merged_boxes[-1] = merge_boxes(merged_boxes[-1], box)
            elif (
                pixels_are_close(box[4], merged_boxes[-1][4])
                and box[2] - merged_boxes[-1][3] < close_threshold
            ):
                if debug:
                    print("close", box[4], merged_boxes[-1][4])
                merged_boxes[-1] = merge_boxes(merged_boxes[-1], box)
            elif (is_tiny_box(box) or is_tiny_box(merged_boxes[-1])) and box[
                2
            ] - merged_boxes[-1][3] < close_threshold:
                pass
            else:
                if debug:
                    print(
                        "not close",
                        box[4],
                        merged_boxes[-1][4],
                        np.sum(np.abs(box[4] - merged_boxes[-1][4])),
                    )
                merged_boxes.append(box)
    return merged_boxes


def get_bounding_boxes(img):
    # Assume this is the cleaned black letter image

    img = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype("int32")

    shape = img.shape[:2]
    visited = np.zeros(shape, dtype=bool)
    bounding_boxes = []  # top bottom left right color

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

    bounding_boxes = [
        box for box in bounding_boxes if box[0] != box[1] and box[2] != box[3]
    ]

    # Merge boxes
    bounding_boxes.sort(key=lambda x: x[2])
    close_threshold = 8
    merged_boxes = merge_all_boxes(
        merge_all_boxes(bounding_boxes, close_threshold=close_threshold),
        close_threshold=close_threshold,
    )

    # Split boxes

    for box in merged_boxes:
        cv2.rectangle(img, (box[2], box[0]), (box[3], box[1]), [255, 0, 0], 1)

    return img, merged_boxes


def resize_with_padding(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
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
    result = np.ones((target_height, target_width), dtype=np.uint8) * 255

    # Compute top-left corner for centering
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # Place resized image onto the white canvas
    result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return result
