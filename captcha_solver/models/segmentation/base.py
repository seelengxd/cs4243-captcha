"""
Base class for segmentation + single character recognition models.
Segmentation accuracy (count is correct):  0.8118962887646162

By implementing this class and implementing fit() and predict() for a model that recognises individual characters,
you can create a new model for segmenting characters from images.
"""

from abc import ABC, abstractmethod
import multiprocessing
import os
import concurrent.futures
import sys
from tqdm import tqdm
import random

import cv2
import numpy as np
from functools import cache

from preprocessing.utils import get_bounding_boxes, resize_with_padding

TRAIN_CHARACTERS_RESIZED = "data/train_cleaned_characters_resized"
TEST_CHARACTERS_RESIZED = "data/test_cleaned_characters_resized"
TEST_FOLDER = "data/test_cleaned_black_resized"

sys.setrecursionlimit(1000000)

cv2.setNumThreads(0)


class SegmentationModelBase(ABC):
    target_image_size: tuple[int, int] = (32, 32)
    images_dict: dict[str, np.ndarray] = {}

    def __init__(self, target_image_size: tuple[int, int]):
        self.target_image_size = target_image_size

    @cache
    def load_images_from_folder(self, folder: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load images from a folder and return them as a tuple of numpy arrays.
        Resize images if target_image_size is set.
        """
        images, labels = [], []
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            for filename in os.listdir(subfolder_path):
                img = cv2.imread(
                    os.path.join(subfolder_path, filename), cv2.IMREAD_GRAYSCALE
                )

                if self.target_image_size:
                    img = cv2.resize(img, self.target_image_size)
                if img is not None:
                    images.append(img.flatten())
                    labels.append(subfolder)
        return np.array(images), np.array(labels)

    def load_test_images(self):
        if not self.images_dict:
            self.images_dict = {
                path: cv2.imread(os.path.join(TEST_FOLDER, path))
                for path in os.listdir(TEST_FOLDER)
                if path.endswith(".png")
            }

    def train(self):
        X_train, y_train = self.load_images_from_folder(TRAIN_CHARACTERS_RESIZED)
        print("Training model...")
        self.fit(X_train, y_train)

    def get_captcha_from_image(self, img):
        _, boxes = get_bounding_boxes(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        letters = [
            resize_with_padding(img[box[0] : box[1], box[2] : box[3]], 169, 78)
            for box in boxes
        ]
        predictions = self.predict(
            [cv2.resize(letter, self.target_image_size).flatten() for letter in letters]
        )
        return len(boxes), "".join(predictions)

    def guess_captcha(self, image: str) -> tuple[tuple[int, int], str, str]:
        img = cv2.imread(os.path.join(TEST_FOLDER, image))
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("something wrong with", image)
            return (0, 0), image, ""

        try:
            _, guess = self.get_captcha_from_image(img)
            correct_answer = image.split("-")[0]
            return (1, 1) if guess == correct_answer else (0, 1), image, guess
        except Exception as e:
            print(e)
            print("something very wrong with", image)
            return (0, 0), image, ""

    def evaluate_characters(self):
        """Return accuracy against test character set."""
        X_test, y_test = self.load_images_from_folder(TEST_CHARACTERS_RESIZED)
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

    def evaluate_captcha(self, parallel=True, limit=None):
        """Return accuracy against test captcha set"""

        images = os.listdir(TEST_FOLDER)
        if limit:
            images = images[:limit]

        if parallel:
            multiprocessing.freeze_support()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.guess_captcha, image) for image in images
                ]

                total = len(futures)
                results = []
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total,
                    desc="Processing images",
                ):
                    results.append(future.result())
        else:
            results = [self.guess_captcha(image) for image in tqdm(images)]

        correct = sum(result[0][0] for result in results)
        length_correct = sum(
            len(image.split("-")[0]) == len(guess) for (_, image, guess) in results
        )
        total = sum(result[0][1] for result in results)
        print(
            f"Segmentation accuracy (count is correct): {length_correct / total} ({length_correct}/{total})"
        )
        print(
            f"Accuracy: {correct / total if total > 0 else 0} ({correct}/{total})",
        )

        with open("predictions.txt", "w") as f:
            for result in results:
                if result[0][0] == 0:
                    f.write(f"{result[1]}: {result[2]}\n")
                else:
                    f.write(f"{result[1]}: {result[2]} (correct)\n")

    def guess_random_captcha(self):
        """Guess a random captcha from the test set."""
        images = os.listdir(TEST_FOLDER)
        image = random.choice(images)
        print(image)
        img = cv2.imread(os.path.join(TEST_FOLDER, image))
        _, guess = self.get_captcha_from_image(img)
        print(guess)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train model for fitting to identify CHARACTERS."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the characters from the given images."""
        raise NotImplementedError
