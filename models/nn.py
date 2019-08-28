import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class MLP(nn.Module):
    def __init__(self, feature_dim, hidsizes, dropout=0., activation='relu'):
        super(MLP, self).__init__()

        if activation == 'relu':
            self.ac_fn = torch.nn.ReLU
        elif activation == 'tanh':
            self.ac_fn = torch.nn.Tanh
        elif activation == 'sigmoid':
            self.ac_fn = torch.nn.Sigmoid
        elif activation == 'leaky':
            self.ac_fn = torch.nn.LeakyReLU
        elif activation == 'elu':
            self.ac_fn = torch.nn.ELU
        elif activation == 'relu6':
            self.ac_fn = torch.nn.ReLU6

        self.mlp = []
        hidsizes = [feature_dim] + hidsizes
        for i in range(1, len(hidsizes)):
            self.mlp.append(nn.Linear(hidsizes[i-1], hidsizes[i]))
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(self.ac_fn())
        self.mlp = nn.Sequential(*self.mlp, nn.Linear(hidsizes[-1], 1))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float)
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        return self.mlp(x).squeeze(-1)

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


class BinaryClassifier(object):
    def __init__(self, model, learning_rate, loss_func='bce'):
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

        if loss_func == 'bce':
            self.transform_y = False
            self.ac_fn = None
            self.loss_func = torch.nn.BCEWithLogitsLoss
        elif loss_func == 'mse':
            self.transform_y = True
            self.ac_fn = torch.tanh
            self.loss_func = torch.nn.MSELoss
        elif loss_func == 'l1':
            self.transform_y = True
            self.ac_fn = torch.tanh
            self.loss_func = torch.nn.L1Loss
        elif loss_func == 'huber':
            self.transform_y = True
            self.ac_fn = torch.tanh
            self.loss_func = torch.nn.SmoothL1Loss
        elif loss_func == 'logistic':
            self.transform_y = True
            self.ac_fn = torch.tanh
            self.loss_func = torch.nn.SoftMarginLoss
        else:
            raise(NotImplementedError, loss_func)

        self.loss = self.loss_func()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    def predict(self, X):
        with torch.no_grad():
            if not self.ac_fn:
                y_pred = torch.sigmoid(self.model(X)).cpu().numpy()  # bce with logits
            else:
                y_pred = self.ac_fn(self.model(X)).cpu().numpy()
        if self.transform_y:
            y_pred[y_pred < 0] = -1
            y_pred[y_pred >= 0] = 1
        else:
            y_pred = y_pred.round()
        return y_pred

    def train(self, X, y):
        self.model.train()

        y_pred = self.model(X)
        if self.ac_fn:
            y_pred = self.ac_fn(y_pred)
        y = torch.tensor(y, dtype=torch.float)

        loss = self.loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def val(self, X, y):
        self.model.eval()
        y_pred = self.predict(torch.tensor(X, dtype=torch.float))
        acc = accuracy_score(y, y_pred)
        return acc

    def fit(self, X_train, y_train, X_val=None, y_val=None, episodes=100, batchsize=None, 
            val_interval=20, log_interval=100, logger=None):
        if self.transform_y:
            y_train[y_train == 0] = -1
            if y_val is not None:
                y_val[y_val == 0] = -1

        train_acc, val_acc, losses = [], [], []
        batchsize = batchsize if batchsize and batchsize < len(X_train) else len(X_train)
        m = X_train.shape[0]

        for ep in range(episodes):
            mb_idxes = np.random.choice(m, batchsize, replace=False)
            mb_X_train, mb_y_train = X_train[mb_idxes], y_train[mb_idxes]
            loss = self.train(mb_X_train, mb_y_train)
            losses.append(loss)
            
            if ep % val_interval == 0 and X_val is not None and y_val is not None:
                train_acc.append(self.val(X_train, y_train))
                val_acc.append(self.val(X_val, y_val))
            if logger is not None and ep % log_interval == 0:
                logger.record_tabular('ep', ep)
                logger.record_tabular('loss', np.mean(losses[-log_interval:]))
                logger.record_tabular('train_acc', np.mean(train_acc[-log_interval//val_interval:]))
                if X_val is not None and y_val is not None:
                    logger.record_tabular('val_acc', np.mean(val_acc[-log_interval//val_interval:]))
                logger.dump_tabular()

        return {'loss': losses, 'train_acc': train_acc, 'val_acc': val_acc}


class SurrogateBinaryClassifier(BinaryClassifier):
    def __init__(self, model, learning_rate, loss_func, e0, e1):
        super(SurrogateBinaryClassifier, self).__init__(model, learning_rate, loss_func)
        self.e = np.array([e0, e1], dtype=float)
        self.loss = self.loss_func(reduction='none')

    def train(self, X, y):
        """ The original surrogate function is:

               (1 - \rho_{-y}) * l(t,y) - \rho_{y} * l(t,-y)
        loss = ---------------------------------------------
                        1 - \rho_{+1} - \rho_{-1}

        where y \in {-1, +1},

        But because we use {0, 1} as the label, so the loss becomes:

               (1 - e_{1-y}) * l(t,y) - e_{y} * l(t,1-y)
        loss = -----------------------------------------
                           1 - e_0 - e_1
        """
        self.model.train()

        y_pred = self.model(X)
        if self.ac_fn:
            y_pred = self.ac_fn(y_pred)
        if self.transform_y:
            y[y == -1] = 0
        c1 = torch.tensor(1 - self.e[np.int32(1-y)], dtype=torch.float)
        c2 = torch.tensor(self.e[np.int32(y)], dtype=torch.float)
        if self.transform_y:
            y[y == 0] = -1
        y = torch.tensor(y, dtype=torch.float)

        loss1 = c1 * self.loss(y_pred, y)
        loss2 = c2 * self.loss(y_pred, -y if self.transform_y else 1 - y)
        loss = torch.mean(loss1 - loss2) / (1 - self.e.sum())
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()


class PeerBinaryClassifier(BinaryClassifier):
    def __init__(self, model, learning_rate, loss_func, alpha=1.):
        super(PeerBinaryClassifier, self).__init__(model, learning_rate, loss_func)
        self.alpha = alpha

    def train(self, X, y, X_, y_):
        self.model.train()

        y_pred = self.model(X)
        if self.ac_fn:
            y_pred = self.ac_fn(y_pred)
        y = torch.tensor(y, dtype=torch.float)

        y_pred_ = self.model(X_)
        if self.ac_fn:
            y_pred_ = self.ac_fn(y_pred_)
        y_ = torch.tensor(y_, dtype=torch.float)

        loss = self.loss(y_pred, y) - self.alpha * self.loss(y_pred_, y_)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def fit(self, X_train, y_train, X_val=None, y_val=None, episodes=100, batchsize=None,
            val_interval=20, log_interval=100, logger=None):
        if self.transform_y:
            y_train[y_train == 0] = -1
            if y_val is not None:
                y_val[y_val == 0] = -1

        losses, train_acc, val_acc = [], [], []
        batchsize = batchsize or len(X_train)
        m = X_train.shape[0]

        for ep in range(episodes):
            mb_idxes = np.random.choice(m, batchsize, replace=False)
            mb_X_train, mb_y_train = X_train[mb_idxes], y_train[mb_idxes]
            mb_X_train_ = X_train[np.random.choice(m, batchsize, replace=False)]
            mb_y_train_ = y_train[np.random.choice(m, batchsize, replace=False)]
            loss = self.train(mb_X_train, mb_y_train, mb_X_train_, mb_y_train_)
            losses.append(loss)
            
            if ep % val_interval == 0 and X_val is not None and y_val is not None:
                train_acc.append(self.val(X_train, y_train))
                val_acc.append(self.val(X_val, y_val))
            if logger is not None and ep % log_interval == 0:
                logger.record_tabular('ep', ep)
                logger.record_tabular('loss', np.mean(losses[-log_interval:]))
                logger.record_tabular('train_acc', np.mean(train_acc[-log_interval//val_interval:]))
                if X_val is not None and y_val is not None:
                    logger.record_tabular('val_acc', np.mean(val_acc[-log_interval//val_interval:]))
                logger.dump_tabular()

        return {
            'loss': losses,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
