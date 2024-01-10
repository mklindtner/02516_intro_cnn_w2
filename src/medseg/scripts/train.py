#!/usr/bin/env python3


import logging
import sys


import torch


from medseg.datasets import PH2Dataset
from medseg.metrics import bce_loss
from medseg.models import EncDec
from medseg.training import train


LOG = logging.getLogger()


DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 6
DEFAULT_CUDA_DEV = 0


def execute_training(
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    cuda_dev=DEFAULT_CUDA_DEV,
):
    torch.cuda.set_device(cuda_dev)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        LOG.info('Using CUDA device %d.', cuda_dev)
    else:
        device = torch.device('cpu')
        LOG.info('Using CPU.')

    trainset = PH2Dataset('train')
    testset = PH2Dataset('test')
    valset = PH2Dataset('validation')

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=3,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
    )

    model = EncDec().to(device)

    LOG.debug('Training for %d epochs.', epochs)
    train(
        model,
        torch.optim.Adam(model.parameters()),
        bce_loss,
        epochs,
        train_loader,
        test_loader,
        device,
    )


def main(argv=sys.argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, metavar='N', help='Number of epochs (default=%(default)s)')
    parser.add_argument('--batch-size', default=DEFAULT_BATCH_SIZE, type=int, metavar='N', help='(Mini-)batch size (default=%(default)s)')
    parser.add_argument('--cuda-dev', default=DEFAULT_CUDA_DEV, type=int, metavar='N', help='CUDA device to use (default=%(default)s)')
    parser.add_argument('--debug', '-v', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args(argv[1:])

    if args.debug:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format='%(name)s: %(message)s')

    execute_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        cuda_dev=args.cuda_dev,
    )


if __name__ == '__main__':
    main()
