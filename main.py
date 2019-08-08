from parser import parse_args
from utils import load_data, equalize_prior, make_noisy_data
from model import BinaryClassifier

from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    dataset = load_data(args.dataset)
    dataset = equalize_prior(dataset)
    X = dataset.drop(['target'], axis=1).values
    y = dataset.target.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_noisy = make_noisy_data(y_train, args.e0, args.e1)
    classifier = BinaryClassifier(X_train.shape[-1], args.lr, args.hidsize)
    train_acc, test_acc = classifier.fit(X_train, y_train, X_test, y_test,
                                         episodes=args.episodes, batchsize=args.batchsize)

    plt.plot(train_acc, label='training curve')
    plt.plot(test_acc, label='testing curve')
    plt.grid()
    plt.legend()
    plt.show()

    print(f'test accuracy is {classifier.test(X_test, y_test)}')


if __name__ == '__main__':
    main()
