# coding: utf-8

"""
Преобразует одномерный тензор, записанный в текстовом виде, в изображение в формате JPG
"""

import argparse as ap
import numpy as np

from PIL import Image


def _setup_parser():
    parser = ap.ArgumentParser()

    parser.add_argument('thensor_path', help='Path to thensor in text format to process')
    parser.add_argument('img_width', type=int, help='Width of image')
    parser.add_argument('--out_path', '-o', type=str, default='result.jpg',
                        help='Path to result image')

    return parser


def main():
    args = _setup_parser().parse_args()

    print('Reading from "{}"...'.format(args.thensor_path))

    try:
        data = np.loadtxt(args.thensor_path, dtype=np.uint8).reshape((-1, args.img_width, 3))
    except IOError:#FileNotFoundError:
        print('Unable to open file "{}"'.format(args.thensor_path))
        return

    print('Saving image to "{}"...'.format(args.out_path))

    try:
        img = Image.fromarray(data)
        img.save(args.out_path)
    except Exception as e:
        print('Error occured: "{}"'.format(str(e)))
        return

    print('Save to "{}" complete'.format(args.out_path))


if __name__ == '__main__':
    main()
