from __future__ import print_function
from sys import argv
import os.path


def otsu(src_path, dst_path):
    pass


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
