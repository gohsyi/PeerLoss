from utils.parser import parse_args


def run(args):
    from run_nn import run as run_nn
    from run_svm import run as run_svm
    from run_pam import run as run_pam

    run_nn(args)
    run_svm(args)
    run_pam(args)


if __name__ == '__main__':
    run(parse_args().__dict__)
