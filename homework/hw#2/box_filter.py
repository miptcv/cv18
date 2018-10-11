from __future__ import print_function
from sys import argv
import os.path


def box_flter(src_path, dst_path, w, h):
    pass


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])
