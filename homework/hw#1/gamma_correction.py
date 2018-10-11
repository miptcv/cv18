from __future__ import print_function
from sys import argv
import os.path


def gamma_correction(src_path, dst_path, a, b):
    pass


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])
