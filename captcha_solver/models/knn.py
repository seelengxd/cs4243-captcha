"""
Baseline Segmentation+KNN model.

Character accuracy: 0.6727502184253956
Captcha accuracy: 0.18962887646161666

"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from models.segmentation.base import SegmentationModelBase


class KNNCaptcha(SegmentationModelBase):
    knn: KNeighborsClassifier

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.knn.predict(X)


if __name__ == "__main__":
    model = KNNCaptcha((32, 32))
    model.train()
    print(model.evaluate_characters())
    model.guess_random_captcha()
    model.evaluate_captcha()
