import numpy as np
from collections import Counter


class KNearestNeighbors:
    """Tiny K-Nearest Neighbors classifier."""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 2))
    y_train = (X_train[:, 0] > 0).astype(int)

    X_test = rng.standard_normal((20, 2))
    y_test = (X_test[:, 0] > 0).astype(int)

    model = KNearestNeighbors(k=3)
    model.fit(X_train, y_train)
    accuracy = (model.predict(X_test) == y_test).mean()
    print(f"accuracy: {accuracy:.4f}")
