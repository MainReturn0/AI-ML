import numpy as np


class LogisticRegression:
    """Tiny logistic regression model trained with gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = self._sigmoid(X @ self.weights + self.bias)
            error = y_pred - y
            self.weights -= self.learning_rate * (1 / n_samples) * (X.T @ error)
            self.bias -= self.learning_rate * (1 / n_samples) * error.sum()

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    accuracy = (model.predict(X) == y).mean()
    print(f"accuracy: {accuracy:.4f}")
