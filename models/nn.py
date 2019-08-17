import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class MLP(nn.Module):
    def __init__(self, feature_dim, hidsize, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidsize),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidsize, hidsize),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidsize, 1)
        )

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float)
        if x.dim() < 2:
            x = x.unsqueeze(0)
        return self.mlp(x).squeeze(-1)

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


class BinaryClassifier(object):
    def __init__(self, feature_dim, learning_rate, hidsize, dropout):
        self.mlp = MLP(feature_dim, hidsize, dropout)
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), learning_rate)

    def predict(self, X):
        self.mlp.eval()
        with torch.no_grad():
            return torch.sigmoid(self.mlp(X)).cpu().numpy().round()

    def train(self, X, y):
        self.mlp.train()
        y_pred = self.mlp(X)
        y = torch.tensor(y, dtype=torch.float)
        loss = self.loss_func(y_pred, y)
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
        batchsize = batchsize or len(X_train)
        m = X_train.shape[0]

        for ep in range(episodes):
            X_train, y_train = shuffle(X_train, y_train)
            for i in range(0, m, batchsize):
                j = min(i + batchsize, m)
                mb_X_train = X_train[i:j]
                mb_y_train = y_train[i:j]
                loss = self.train(mb_X_train, mb_y_train)

            train_acc.append(self.test(X_train, y_train))
            if X_test is not None and y_test is not None:
                test_acc.append(self.test(X_test, y_test))

        return train_acc, test_acc


class SurrogateBinaryClassifier(BinaryClassifier):
    def __init__(self, feature_dim, learning_rate, hidsize, dropout, e0, e1):
        super(SurrogateBinaryClassifier, self).__init__(feature_dim, learning_rate, hidsize, dropout)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.e = np.array([e0, e1], dtype=float)

    def train(self, X, y):
        """ The original surrogate function is:

               (1 - \rho_{-y}) * l(t,y) - \rho_{y} * l(t,-y)
        loss = ---------------------------------------------
                        1 - \rho_{+1} - \rho_{-1}

        where y \in {-1, +1},

        But because we use {0, 1} as the label, so the loss becomes:

               (1 - e_{1-y}) * l(t,y) - e_{y} * l(t,1-y)
        loss = -----------------------------------------
                        1 - e_{+1} - e_{-1}
        """
        self.mlp.train()
        y_pred = self.mlp(X)
        c1 = torch.tensor(1 - self.e[[int(1-yy) for yy in y]], dtype=torch.float)
        c2 = torch.tensor(self.e[[int(yy) for yy in y]], dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        loss1 = c1 * self.loss_func(y_pred, y)
        loss2 = c2 * self.loss_func(y_pred, 1 - y)
        loss = torch.mean((loss1 - loss2) / (1 - self.e.sum()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class PeerBinaryClassifier(BinaryClassifier):
    def __init__(self, feature_dim, learning_rate, hidsize, dropout, alpha):
        super(PeerBinaryClassifier, self).__init__(feature_dim, learning_rate, hidsize, dropout)
        self.alpha = alpha

    def train(self, X, y, X_, y_):
        self.mlp.train()

        y_pred = self.mlp(X)
        y = torch.tensor(y, dtype=torch.float)
        y_pred_ = self.mlp(X_)
        y_ = torch.tensor(y_, dtype=torch.float)
        loss = self.loss_func(y_pred, y) - self.alpha * self.loss_func(y_pred_, y_)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X_train, y_train, X_test=None, y_test=None, episodes=100, batchsize=None):
        train_acc, test_acc = [], []
        batchsize = batchsize or len(X_train)
        m = X_train.shape[0]

        for ep in range(episodes):
            X_train, y_train = shuffle(X_train, y_train)
            for i in range(0, m, batchsize):
                j = min(i + batchsize, m)
                mb_X_train = X_train[i:j]
                mb_y_train = y_train[i:j]
                mb_X_train_ = X_train[np.random.choice(m, batchsize)]
                mb_y_train_ = y_train[np.random.choice(m, batchsize)]
                loss = self.train(mb_X_train, mb_y_train, mb_X_train_, mb_y_train_)

            train_acc.append(self.test(X_train, y_train))
            if X_test is not None and y_test is not None:
                test_acc.append(self.test(X_test, y_test))

        return train_acc, test_acc
