"""
Baseline Segmentation+CNN model.

Character accuracy: 0.745655761576546
Captcha accuracy: 0.25673614641586173

References: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import os
import pickle
import numpy as np

from models.segmentation.base import SegmentationModelBase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNCaptcha(SegmentationModelBase):
    net: Net
    LABELS = "0123456789abcdefghijklmnopqrstuvwxyz"
    EPOCH_COUNT = 80

    def _transform_x(self, X: np.ndarray) -> torch.Tensor:
        N = X.shape[0]  # Number of images
        H, W, C = 32, 32, 1  # Expected image dimensions
        X = X.reshape(N, H, W, C)  # Reshape from (N, 1024) to (N, 32, 32, 3)

        # Now perform the transpose
        X = X.transpose((0, 3, 1, 2))  # Convert from (N, H, W, C) to (N, C, H, W)

        # Convert to float tensor and normalize to [-1, 1]
        X_tensor = torch.FloatTensor(X) / 255.0  # Normalize pixel values
        return X_tensor

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Convert labels to numerical indices
        y = np.array([self.LABELS.index(label) for label in y])

        # Reshape X to PyTorch format: (N, C, H, W)
        print("X shape before reshaping:", X.shape)

        X_tensor = self._transform_x(X)
        y_tensor = torch.LongTensor(y)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.EPOCH_COUNT):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data

                # Move tensors to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs, labels = inputs.to(device), labels.to(device)
                self.net.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                # Print loss every 200 batches
                running_loss += loss.item()
                if i % 200 == 199:
                    print(
                        f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.3f}"
                    )
                    running_loss = 0.0

        print("Finished Training")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X)
        predictions = self.net(self._transform_x(X))

        # Convert tensor predictions to indices
        predicted_indices = (
            predictions.argmax(dim=1).cpu().numpy()
        )  # Ensure it's a NumPy array

        return np.array(
            [self.LABELS[pred] for pred in predicted_indices]
        )  # Index properly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", default="segmentation_and_cnn_model.pkl", required=False
    )
    args = parser.parse_args()
    filename = args.filename

    if os.path.exists(filename):
        with open(filename, "rb") as fr:
            model = pickle.load(fr)
    else:
        model = CNNCaptcha((32, 32))
        model.train()
        with open(filename, "wb") as fw:
            pickle.dump(model, fw)

    print(model.evaluate_characters())
    model.guess_random_captcha()
    model.evaluate_captcha()
