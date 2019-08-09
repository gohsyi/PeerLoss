import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


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
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float)
        if x.dim() < 2:
            x = x.unsqueeze(0)
        return self.mlp(x).squeeze()

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


class BinaryClassifier:
    def __init__(self, feature_dim, lr, hidsize):
        self.mlp = MLP(feature_dim, hidsize)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr)

    def predict(self, X):
        with torch.no_grad():
            return torch.sigmoid(self.mlp(X)).cpu().numpy().round()

    def train(self, X, y, X_=None, y_=None):
        y_pred = self.mlp(X)
        y = torch.tensor(y, dtype=torch.float)
        loss = self.criterion(y_pred, y)
        if X_ is not None and y_ is not None:
            y_pred_ = self.mlp(X_)
            y_ = torch.tensor(y_, dtype=torch.float)
            loss -= self.criterion(y_pred_, y_)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, X, y):
        y_pred = self.predict(torch.tensor(X, dtype=torch.float))
        acc = accuracy_score(y, y_pred)
        return acc

    def fit(self, X_train, y_train, X_test=None, y_test=None, episodes=100, batchsize=None, peer_loss=False):
        train_acc, test_acc = [], []
        batchsize = batchsize or len(X_train)
        m = X_train.shape[0]

        for ep in range(episodes):
            X_train, y_train = shuffle(X_train, y_train)
            for i in range(0, m, batchsize):
                j = min(i + batchsize, m)
                mb_X_train = X_train[i:j]
                mb_y_train = y_train[i:j]
                if peer_loss:
                    random_idxes = np.random.choice(m, batchsize)
                    mb_X_train_ = X_train[random_idxes]
                    mb_y_train_ = y_train[random_idxes]
                    loss = self.train(mb_X_train, mb_y_train, mb_X_train_, mb_y_train_)
                loss = self.train(mb_X_train, mb_y_train)

            train_acc.append(self.test(X_train, y_train))
            if X_test is not None and y_test is not None:
                test_acc.append(self.test(X_test, y_test))

        return train_acc, test_acc
