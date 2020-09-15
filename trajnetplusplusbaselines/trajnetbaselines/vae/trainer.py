"""Command line tool to train a VAE model."""

import argparse

class Trainer():
    def __init__(self):
        raise NotImplementedError

def main(epochs=50):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
        help='number of epochs')


if __name__ == '__main__':
    main()
