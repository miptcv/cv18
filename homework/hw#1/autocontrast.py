from __future__ import print_function
from sys import argv
import os.path


def autocontrast(src_path, dst_path, white_perc, black_perc):
    pass


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] < 1
    assert 0 <= argv[4] < 1

    autocontrast(*argv[1:])
