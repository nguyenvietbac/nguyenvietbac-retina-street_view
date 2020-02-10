import argparse
import os
import sys
import warnings

# import keras
# import keras.preprocessing.image
# import tensorflow as tf

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from ..custom_slide_window.csv_generator import CSVGenerator


def main():
    print('None')


if __name__ == '__main__':
    main()
