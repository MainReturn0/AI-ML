import numpy as np


class NeuralNetwork:
    """Tiny single-hidden-layer neural network for binary classification."""

    def __init__(self, hidden_size=8, learning_rate=0.01, n_iterations=1000):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.params = {}

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(0)

        self.params = {
            "W1": rng.standard_normal((n_features, self.hidden_size)) * 0.01,
            "b1": np.zeros(self.hidden_size),
            "W2": rng.standard_normal((self.hidden_size, 1)) * 0.01,
            "b2": np.zeros(1),
        }

        for _ in range(self.n_iterations):
            # Forward pass
            Z1 = X @ self.params["W1"] + self.params["b1"]
            A1 = self._relu(Z1)
            Z2 = A1 @ self.params["W2"] + self.params["b2"]
            A2 = self._sigmoid(Z2).squeeze()

            # Backward pass
            dA2 = -(y / (A2 + 1e-8) - (1 - y) / (1 - A2 + 1e-8)) / n_samples
            dZ2 = dA2 * A2 * (1 - A2)
            dW2 = A1.T @ dZ2.reshape(-1, 1)
            db2 = dZ2.sum()

            dA1 = dZ2.reshape(-1, 1) @ self.params["W2"].T
            dZ1 = dA1 * (Z1 > 0)
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)

            self.params["W1"] -= self.learning_rate * dW1
            self.params["b1"] -= self.learning_rate * db1
            self.params["W2"] -= self.learning_rate * dW2
            self.params["b2"] -= self.learning_rate * db2

    def predict_proba(self, X):
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = self._relu(Z1)
        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        return self._sigmoid(Z2).squeeze()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int)

    model = NeuralNetwork(hidden_size=8, learning_rate=0.1, n_iterations=2000)
    model.fit(X, y)
    accuracy = (model.predict(X) == y).mean()
    print(f"accuracy: {accuracy:.4f}")
