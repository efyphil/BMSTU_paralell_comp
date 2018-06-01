# coding: utf-8

"""
Преобразует изображение в формате JPG в одномерный тензор,
записываемый в текстовом виде, то есть

1 2 3 4 5 6 7 ...

где '1 2 3' - это компоненты R, G, B (соответственно) первого пикселя и т.д.
"""

import argparse as ap
import numpy as np

from PIL import Image


def _setup_parser():
    parser = ap.ArgumentParser()

    parser.add_argument('image_path', help='Path to image to process.')
    parser.add_argument('--out_path', '-o', default='out.txt',
                        help='Path to output file. If exists, will be overwritten')

    return parser


def main():
    args = _setup_parser().parse_args()

    try:
        img = Image.open(args.image_path)
    except IOError:
        print('Unable to open file "{}"'.format(args.image_path))
        return

    print('Saving image to "{}"... Please, wait.'.format(args.out_path))

    # Берем данные из объекта PIL.Image, преобразуем их в iterable (list),
    # затем в np.array с типом данных uint8 (как unsigned char),
    # снижаем размерность до 1
    img_data = np.array(list(img.getdata())).astype(np.uint8).flatten()
    np.savetxt(args.out_path, img_data, fmt='%u ')

    print('Save to "{}" complete.'.format(args.out_path))


if __name__ == '__main__':
    main()
