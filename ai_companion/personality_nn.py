"""
Small, efficient neural network implemented with numpy for personality modeling.
This MLP processes a fixed-size embedding + behavioral features vector and
predicts a compact personality vector (e.g., 8 dimensions) representing
openness, conscientiousness, extraversion, agreeableness, neuroticism-like traits,
plus humor and optimism scores.
"""
import numpy as np
from typing import Tuple


class SimpleMLP:
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 8, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0, 0.1, (hidden_size, input_size))
        self.b1 = np.zeros((hidden_size,))
        self.W2 = rng.normal(0, 0.1, (output_size, hidden_size))
        self.b2 = np.zeros((output_size,))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _softclip(x):
        # outputs in [0,1] approximate
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self._relu(x @ self.W1.T + self.b1)
        out = self._softclip(h @ self.W2.T + self.b2)
        return out

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single gradient step using mean squared error. Returns loss.
        x: (batch, input)
        y: (batch, output)
        """
        batch = x.shape[0]
        # forward
        hpre = x @ self.W1.T + self.b1  # (batch, hidden)
        h = self._relu(hpre)
        outpre = h @ self.W2.T + self.b2
        out = self._softclip(outpre)

        loss = float(np.mean((out - y) ** 2))

        # backward (approx using jacobian of softclip ~ sigmoid'*1)
        dout = 2 * (out - y) / batch
        sigma = out * (1 - out)  # derivative of sigmoid-like
        doutpre = dout * sigma  # (batch, output)

        dW2 = doutpre.T @ h  # (output, hidden)
        db2 = np.sum(doutpre, axis=0)

        dh = doutpre @ self.W2  # (batch, hidden)
        dhpre = dh * (hpre > 0).astype(float)  # relu grad

        dW1 = dhpre.T @ x  # (hidden, input)
        db1 = np.sum(dhpre, axis=0)

        # gradient step
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return loss