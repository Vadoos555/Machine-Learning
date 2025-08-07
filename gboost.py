import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        min_error = float("inf")

        for feature_i in range(n_features):
            X_column = X[:, feature_i]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_mask = X_column <= threshold
                right_mask = X_column > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_value = np.mean(y[left_mask])
                right_value = np.mean(y[right_mask])

                y_pred = np.where(left_mask, left_value, right_value)
                error = np.mean((y - y_pred) ** 2)

                # # store the best configuration
                if error < min_error:
                    min_error = error
                    self.feature_idx = feature_i
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        feature = X[:, self.feature_idx]
        return np.where(feature <= self.threshold, self.left_value, self.right_value)


class GradientBoost:
    def __init__(self, n_estimators=5, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.initial_pred = 0

    def fit(self, X, y):
        # initiaze
        self.initial_pred = np.mean(y)
        y_pred = np.full_like(y, self.initial_pred, dtype=np.float64)

        for _ in range(self.n_estimators):
            # calculate residuals
            residuals = y - y_pred

            stump = DecisionStump()
            stump.fit(X, residuals)

            update = stump.predict(X)

            # update prediction
            y_pred += self.learning_rate * update
            self.models.append(stump)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_pred, dtype=np.float64)

        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)

        return np.where(y_pred >= 0, 1, -1)
    