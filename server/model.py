import os
import numpy as np

import tensorflow as tf

from layers import conv_layer
from utils import init_weights, gen_data


class CryptoNet(object):
    def __init__(self, sess, model_path, msg_len=96):
        self.sess = sess
        self.model_path = model_path
        self.msg_len = msg_len
        self.key_len = self.msg_len
        self.N = self.msg_len

        self.build_model()
        self.load_model()

    def build_model(self):
        # Weights for fully connected layers
        self.w_alice = init_weights("alice_w", [2 * self.N, 2 * self.N])

        # Placeholder variables for Message and Key
        self.msg = tf.placeholder("float", [None, self.msg_len])
        self.key = tf.placeholder("float", [None, self.key_len])

        # Alice's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.alice_input = tf.concat([self.msg, self.key], 1)
        self.alice_hidden = tf.nn.sigmoid(tf.matmul(self.alice_input, self.w_alice))
        self.alice_hidden = tf.expand_dims(self.alice_hidden, 2)
        self.alice_output = tf.squeeze(conv_layer(self.alice_hidden, "alice"))

    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)
        print('model restored')

    def encrypt_hex(self, msg: list, key: list):
        assert len(msg) == self.msg_len, \
            'illegal msg (assert len(msg) == {})'.format(self.msg_len)
        assert len(key) == self.key_len, \
            'illegal key (assert len(key) == {})'.format(self.key_len)

        for i, j in zip(msg, key):
            assert not (i < -1 or i > 1 or j < -1 or j > 1), \
                'values should be in the range of [-1, 1]'

        msg = np.array(msg).reshape(-1, self.msg_len)
        key = np.array(key).reshape(-1, self.key_len)
        return self.sess.run(self.alice_output,
                             feed_dict={self.msg: msg, self.key: key})

    def test_interactive(self):
        def convert(x):
            return np.array([[2*int(i)-1 for i in x]])

        while True:
            P = convert(input('MSG> ')[:self.msg_len])
            K = convert(input('KEY> ')[:self.key_len])

            enc = self.sess.run(
                self.alice_output,
                feed_dict={self.msg: P, self.key: K})
            print('ENC>', ''.join(np.where(enc>0, '1', '0')))
