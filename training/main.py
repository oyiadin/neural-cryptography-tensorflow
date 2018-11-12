################################
# original author: @ankeshanand
# repo: https://github.com/ankeshanand/neural-cryptography-tensorflow
# slightly modified by @oyiadin
################################


import tensorflow as tf

from argparse import ArgumentParser
from src.model import CryptoNet


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='message length',
                        metavar='MSG_LEN', default=96)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=0.0008)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=50)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=1024)

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    with tf.Session() as sess:
        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)

        crypto_net.train()
        crypto_net.test_interactive()


if __name__ == '__main__':
    main()
