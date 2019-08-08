import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['heart'], type=str, default='heart')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--e0', type=float, default=0.2)
    parser.add_argument('--e1', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidsize', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=10)
    args = parser.parse_args()
    return args
