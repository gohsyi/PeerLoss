import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class MLP(nn.Module):
    def __init__(self, feature_dim, hidsize):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidsize),
            nn.ReLU(),
            nn.Linear(hidsize, hidsize),
            nn.ReLU(),
            nn.Linear(hidsize, 1)
        )

    def forward(self, x):
        return self.mlp(x)


class BinaryClassifier:
    def __init__(self, feature_dim, lr, hidsize):
        self.mlp = MLP(feature_dim, hidsize)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr)

    def predict(self, X):
        with torch.no_grad():
            return torch.sigmoid(self.mlp(X)).cpu().numpy().round()

    def train(self, X, y):
        y_pred = self.mlp(X)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, X, y):
        y_pred = self.predict(torch.tensor(X, dtype=torch.float))
        acc = accuracy_score(y, y_pred)
        return acc

    def fit(self, X_train, y_train, X_test=None, y_test=None, episodes=100, batchsize=None):
        train_acc, test_acc = [], []
        y_train = y_train[..., np.newaxis]
        y_test = y_test[..., np.newaxis]
        batchsize = batchsize or len(X_train)

        for ep in range(episodes):
            for i in range(0, len(X_train), batchsize):
                j = min(i + batchsize, len(X_train))
                X_train_ = torch.tensor(X_train[i:j], dtype=torch.float)
                y_train_ = torch.tensor(y_train[i:j], dtype=torch.float)
                loss = self.train(X_train_, y_train_)

            train_acc.append(self.test(X_train, y_train))
            if X_test is not None and y_test is not None:
                test_acc.append(self.test(X_test, y_test))

        return train_acc, test_acc
