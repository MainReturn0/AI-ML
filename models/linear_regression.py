import numpy as np


class LinearRegression:
    """Tiny linear regression model trained with gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            self.weights -= self.learning_rate * (2 / n_samples) * (X.T @ error)
            self.bias -= self.learning_rate * (2 / n_samples) * error.sum()

    def predict(self, X):
        return X @ self.weights + self.bias


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1))
    y = 3 * X[:, 0] + 2 + rng.standard_normal(100) * 0.1

    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    print(f"weight: {model.weights[0]:.4f}, bias: {model.bias:.4f}")
