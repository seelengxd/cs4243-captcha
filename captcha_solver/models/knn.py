
import os
import multiprocessing
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from preprocessing.utils import color_letters_black, get_bounding_boxes, resize_with_padding

# Set OpenCV to single-threaded mode to avoid conflicts with multiprocessing
cv2.setNumThreads(0)

TRAIN_CHARACTERS_RESIZED = "data/train_cleaned_characters_resized"
TEST_FOLDER = "data/test_cleaned_black_resized"

SIZE = (32, 32)
def load_images_from_folder(folder):
    images, labels = [], []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img = cv2.imread(os.path.join(subfolder_path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, SIZE)
            if img is not None:
                images.append(img.flatten())
                labels.append(subfolder)
    return np.array(images), np.array(labels)

X_train, y_train = load_images_from_folder(TRAIN_CHARACTERS_RESIZED)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

def get_captcha_from_image(img):
    _, boxes = get_bounding_boxes(img)
    # img = color_letters_black(img)
    letters = [resize_with_padding(img[box[0]:box[1], box[2]:box[3]], 169, 78) for box in boxes]
    predictions = knn.predict([cv2.resize(letter, SIZE).flatten() for letter in letters])
    return len(boxes), "".join(predictions)

images_dict = {
    path: cv2.imread(os.path.join(TEST_FOLDER, path), cv2.IMREAD_GRAYSCALE)
    for path in os.listdir(TEST_FOLDER) if path.endswith(".png")
}

def process_image(args):
    image = args
    path = os.path.join(TEST_FOLDER, image)
    img = images_dict.get(image)
    if img is None:
        return (0, 0), image, ""
    
    _, guess = get_captcha_from_image(img)
    correct_answer = image.split('-')[0]
    return (1, 1) if guess == correct_answer else (0, 1), image, guess

if __name__ == "__main__":
    multiprocessing.freeze_support()
    images = os.listdir(TEST_FOLDER)
    num_workers = os.cpu_count()
    
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, image)
            for image in images
        ]

        from tqdm import tqdm
        total = len(futures)
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=total, desc="Processing images"):
            results.append(future.result())

    correct = sum(result[0][0] for result in results)
    total = sum(result[0][1] for result in results)
    print("Accuracy:", correct / total if total > 0 else 0)

    with open("predictions.txt", "w") as f:
        for result in results:
            if result[0][0] == 0:
                f.write(f"{result[1]}: {result[2]}\n")
            else:
                f.write(f"{result[1]}: {result[2]} (correct)\n")
