#!/usr/bin/env python3


import logging
from pathlib import Path
from random import shuffle


LOG = logging.getLogger()


def save_list(path, l):
    with Path(path).open('xt') as file:
        file.writelines([f'{n}\n' for n in l])


def index_from_ratio(total, ratio):
    return int(ratio * total)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--first', '-f',
        metavar='N',
        type=int,
        default=21,
        help='default %(default)d',
    )
    parser.add_argument(
        '--last', '-l',
        metavar='N',
        type=int,
        default=40,
        help='default %(default)d',
    )
    parser.add_argument(
        '--val',
        metavar='RATIO',
        type=float,
        default=0.2,
        help='default %(default).1f',
    )
    parser.add_argument(
        '--test',
        metavar='RATIO',
        type=float,
        default=0.2,
        help='default %(default).1f',
    )
    parser.add_argument(
        '--output-path',
        metavar='PATH',
        type=Path,
        default='.',
        help='default "%(default)s"',
    )
    parser.add_argument(
        '--debug', '-v',
        action='store_true',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
    )
    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARN
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format='%(message)s')

    # End of setup.

    numbers = list(range(args.first, args.last+1))
    shuffle(numbers)

    total = len(numbers)
    LOG.debug('Total number of datapoints: %d', total)

    begin_test = index_from_ratio(total, args.val)
    begin_train = index_from_ratio(total, args.val + args.test)
    LOG.debug(
        'Picking %d validation samples and %d test samples.',
        begin_test,
        begin_train-begin_test,
    )

    indices = [
        ('validation', (0, begin_test)),
        ('test', (begin_test, begin_train)),
        ('train', (begin_train, total)),
    ]

    for name, (begin, end) in indices:
        path = args.output_path / f'{name}.txt'
        LOG.info('Saving %d "%s" indices into "%s".', end-begin, name, path)

        save_list(path, numbers[begin:end])


if __name__ == '__main__':
    main()
