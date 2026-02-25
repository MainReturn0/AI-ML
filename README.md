# AI-ML

A collection of tiny AI/ML models implemented from scratch using NumPy.

## Models

| Model | File | Task |
|-------|------|------|
| Linear Regression | `models/linear_regression.py` | Regression |
| Logistic Regression | `models/logistic_regression.py` | Binary classification |
| K-Nearest Neighbors | `models/knn.py` | Classification |
| Neural Network | `models/neural_network.py` | Binary classification |

## Usage

```python
import numpy as np
from models import LinearRegression, LogisticRegression, KNearestNeighbors, NeuralNetwork

# Linear Regression
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Logistic Regression
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# K-Nearest Neighbors
model = KNearestNeighbors(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Neural Network (single hidden layer)
model = NeuralNetwork(hidden_size=8, learning_rate=0.1, n_iterations=2000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Requirements

```
pip install -r requirements.txt
```