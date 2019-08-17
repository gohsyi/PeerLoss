"""https://github.com/lightgan/numpy-perceptron/blob/develop/perceptron.ipynb"""
import numpy as np


class Perceptron(object):
    """
    This is a perceptron class that can be used for all different variants of perceptrons
    depending on how it is initialized.
    """

    # Decaying learning rate, using margin, using average, and using a seed is all optional.
    def __init__(self, learning_rate, decay_lr=False, margin=0.0, average=False, aggresive=False):
        # Initialize needed variables
        self.W = np.divide((np.random.rand(1, 19) * 2) - 1, 100)
        self.b = ((np.random.rand() * 2) - 1) / 100
        self.lr = learning_rate

        # Learning rate decay
        self.initial_lr = learning_rate
        self.decay_learning = decay_lr
        self.step = 0

        # Margin for mistakes
        self.margin = margin

        # Averaged
        self.average = average
        if self.average:
            self.a = np.copy(self.W)
            self.ba = np.copy(self.b)

        # Aggressive update with margin
        self.aggressive = aggresive

    def train(self, data, labels, epochs=10, dev_x=None, dev_y=None, updates=False):
        self.step = 0
        pred_acc = []
        dev_acc = []
        num_updates = 0
        updates_list = []

        for i in range(epochs):
            # Shuffle data for this epoch
            permutation = np.random.permutation(data.shape[0])

            X = data[permutation]
            y = labels[permutation]

            # Predict and update if needed
            if updates:
                pred, temp_updates = self.predict_and_update(X, y, updates=updates)
                # Update number of updates
                num_updates += temp_updates
                updates_list.append(temp_updates)
            else:
                pred = self.predict_and_update(X, y)

            pred_acc.append(np.array(y == pred).sum() / len(y))

            # If testing on dev set then evaluate
            if dev_x is not None:
                fold_predictions = self.predict(dev_x)
                fold_correct = (fold_predictions == dev_y).sum()

                dev_acc.append(fold_correct / len(fold_predictions))

        if dev_x is not None:
            if updates:
                return pred_acc, dev_acc, updates_list
            # Return prediction and dev acc
            return pred_acc, dev_acc
        else:
            # Return only prediction accuracy
            return pred_acc

    def predict_and_update(self, x, y, updates=False):
        num_updates = 0
        pred = np.zeros((len(y)))

        for i, example in enumerate(x):

            single_pred = self.predict_single(example)

            # Set label depending on sign of pred
            if single_pred >= 0:
                pred[i] = 1
            else:
                pred[i] = -1

            # Check if a mistake was made or we are below the margin
            if (pred[i] != y[i]) or (y[i] * single_pred < self.margin):
                num_updates += 1
                if self.aggressive:
                    learning_rate = ((self.margin - (y[i] * single_pred)) / (np.dot(example.T, example) + 1))
                    # sign of y determines if we add or subtract
                    self.W = self.W + (learning_rate * x[i] * y[i])
                    self.b = self.b + (learning_rate * y[i])
                else:
                    # sign of y determines if we add or subtract
                    self.W = self.W + (self.lr * x[i] * y[i])
                    self.b = self.b + (self.lr * y[i])

            # Update averaged weight vector
            if self.average:
                self.a = self.a + self.W
                self.ba = self.ba + self.b

            # Decay learning rate if decaying
            if self.decay_learning:
                self.lr = self.initial_lr / (1 + self.step)
                self.step = self.step + 1

        if updates:
            return pred, num_updates
        else:
            return pred

    def predict_single(self, x):
        return np.dot(self.W, x.T).flatten() + self.b

    def predict_single_average(self, x):
        return np.dot(self.a, x.T).flatten() + self.ba

    def predict(self, x):
        # If average perceptron then use average
        if self.average:
            res = self.predict_single_average(x)
        else:
            res = self.predict_single(x)

        for i in range(len(res)):
            if res[i] >= 0:
                res[i] = 1
            else:
                res[i] = -1

        return res
