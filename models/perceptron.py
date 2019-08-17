"""https://github.com/lightgan/numpy-perceptron/blob/develop/perceptron.ipynb"""
import numpy as np
from sklearn.utils import shuffle


class Perceptron(object):
    """
    This is a perceptron class that can be used for all different variants of perceptrons
    depending on how it is initialized.
    """
    def __init__(self, feature_dim, learning_rate=0.001, margin=0.0):
        # Initialize needed variables
        self.W = np.random.uniform(-0.1, 0.1, size=(1, feature_dim))
        self.b = np.random.uniform(-0.01, 0.01)
        self.lr = learning_rate
        self.margin = margin

    def fit(self, X_train, y_train, epochs=1000):
        pred_acc = []
        updates_list = []
        y_train[y_train == 0] = -1

        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            # Predict and update if needed
            y_pred, updates = self._predict_and_update(X_train, y_train)
            # Update number of updates
            updates_list.append(updates)
            pred_acc.append(np.mean(y_train == y_pred))

        print(f'margin:{self.margin}\ttrain_acc:{np.mean(pred_acc[-100:])}')
        return pred_acc

    def _predict_and_update(self, X, y):
        num_updates = 0
        y_pred = np.zeros((len(y)))
        for i, x in enumerate(X):
            single_pred = self._predict(x)
            # Set label depending on sign of pred
            y_pred[i] = 1 if single_pred >= 0 else -1
            # Check if a mistake was made or we are below the margin
            if (y_pred[i] != y[i]) or (y[i] * single_pred < self.margin):
                num_updates += 1
                # sign of y determines if we add or subtract
                self.W = self.W + (self.lr * X[i] * y[i])
                self.b = self.b + (self.lr * y[i])
        return y_pred, num_updates

    def _predict(self, x):
        return np.dot(self.W, x.T).flatten() + self.b

    def predict(self, x):
        res = self._predict(x)
        res[res >= 0] = 1
        res[res < 0] = 0
        return res

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
