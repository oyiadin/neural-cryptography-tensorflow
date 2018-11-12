import os
import datetime
import tensorflow as tf
import numpy as np

from .layers import conv_layer
from .config import *
from .utils import init_weights, gen_data


class CryptoNet(object):
    def __init__(self, sess, msg_len=MSG_LEN, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        """
        Args:
            sess: TensorFlow session
            msg_len: The length of the input message to encrypt.
            key_len: Length of Alice and Bob's private key.
            batch_size: Minibatch size for each adversarial training
            epochs: Number of epochs in the adversarial training
            learning_rate: Learning Rate for Adam Optimizer
        """

        self.sess = sess
        self.msg_len = msg_len
        self.key_len = self.msg_len
        self.N = self.msg_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.build_model()
        self.restore_model()

    def build_model(self):
        # Weights for fully connected layers
        self.w_alice = init_weights("alice_w", [2 * self.N, 2 * self.N])
        self.w_bob = init_weights("bob_w", [2 * self.N, 2 * self.N])
        self.w_eve1 = init_weights("eve_w1", [self.N, 2 * self.N])
        self.w_eve2 = init_weights("eve_w2", [2 * self.N, 2 * self.N])

        # Placeholder variables for Message and Key
        self.msg = tf.placeholder("float", [None, self.msg_len])
        self.key = tf.placeholder("float", [None, self.key_len])

        # Alice's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.alice_input = tf.concat([self.msg, self.key], 1)
        self.alice_hidden = tf.nn.sigmoid(tf.matmul(self.alice_input, self.w_alice))
        self.alice_hidden = tf.expand_dims(self.alice_hidden, 2)
        self.alice_output = tf.squeeze(conv_layer(self.alice_hidden, "alice"))

        # Bob's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.bob_input = tf.concat(
            [tf.reshape(self.alice_output, (-1, self.msg_len)), self.key], 1,
            name='bob_input')
        self.bob_hidden = tf.nn.sigmoid(tf.matmul(self.bob_input, self.w_bob))
        self.bob_hidden = tf.expand_dims(self.bob_hidden, 2)
        self.bob_output = tf.squeeze(conv_layer(self.bob_hidden, "bob"))

        # Eve's network
        # FC layer -> FC layer -> Conv Layer (4 1-D convolutions)
        self.eve_input = tf.reshape(self.alice_output, (-1, self.msg_len))
        self.eve_hidden1 = tf.nn.sigmoid(tf.matmul(self.eve_input, self.w_eve1))
        self.eve_hidden2 = tf.nn.sigmoid(tf.matmul(self.eve_hidden1, self.w_eve2))
        self.eve_hidden2 = tf.expand_dims(self.eve_hidden2, 2)
        self.eve_output = tf.squeeze(conv_layer(self.eve_hidden2, "eve"))

        # Loss Functions
        self.decrypt_err_eve = (0.5 - tf.reduce_mean(tf.abs(self.msg - self.eve_output))) ** 2.
        self.decrypt_err_bob = tf.reduce_mean(tf.abs(self.msg - self.bob_output))
        self.loss_bob = self.decrypt_err_bob + self.decrypt_err_eve
        # I managed to modify the loss functions to improve the performance

        # logging
        self.writer = tf.summary.FileWriter(
            "logs", filename_suffix=str(datetime.datetime.now()))
        tf.summary.scalar("eve_err", self.decrypt_err_eve)
        tf.summary.scalar("bob_err", self.decrypt_err_bob)
        self.merged_summary = tf.summary.merge_all()
        self.writer.add_graph(self.sess.graph)

        # Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice_' in var.name or 'bob_' in var.name]
        self.alice_or_eve_vars = [var for var in self.t_vars if 'alice_' in var.name or 'eve_' in var.name]

        # Build the optimizers
        self.bob_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_bob, var_list=self.alice_or_bob_vars)
        self.eve_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.decrypt_err_eve, var_list=self.alice_or_eve_vars)

    def restore_model(self):
        # restore or initialize
        if not os.path.isfile(os.path.join('saved-model', 'checkpoint')):
            tf.global_variables_initializer().run()
            print('variables initialized')
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join('saved-model', 'alice_bob'))
            print('model restored from saved-model/alice_bob')

    def train(self):
        # Begin Training
        for i in range(self.epochs):
            iterations = 100

            print('Training Alice and Bob, Epoch:', i + 1)
            bob_loss, _ = self._train('bob', iterations)
            print('Training Eve, Epoch:', i + 1)
            eve_loss, _ = self._train('eve', iterations)

            P, K = gen_data(
                n=self.batch_size, msg_len=self.msg_len, key_len=self.key_len)
            self.writer.add_summary(
                self.sess.run(self.merged_summary,
                              feed_dict={self.msg: P, self.key: K}),
                global_step=i)
            self.writer.flush()

            # save session
            if not os.path.isdir('saved-model'):
                os.makedirs('saved-model')
            saver = tf.train.Saver()
            saver.save(self.sess, os.path.join('saved-model', 'alice_bob'))
            print('model saved to saved-model/alice_bob')

    def _train(self, network, iterations):
        bob_decrypt_error, eve_decrypt_error = 1., 1.

        bs = self.batch_size
        # Train Eve for two minibatches to give it a slight computational edge
        if network == 'eve':
            bs *= 2

        for i in range(iterations):
            msg_in_val, key_val = gen_data(n=bs, msg_len=self.msg_len, key_len=self.key_len)

            if network == 'bob':
                _, decrypt_err = self.sess.run(
                    [self.bob_optimizer, self.decrypt_err_bob],
                    feed_dict={self.msg: msg_in_val, self.key: key_val})
                bob_decrypt_error = min(bob_decrypt_error, decrypt_err)

            elif network == 'eve':
                _, decrypt_err = self.sess.run(
                    [self.eve_optimizer, self.decrypt_err_eve],
                    feed_dict={self.msg: msg_in_val, self.key: key_val})
                eve_decrypt_error = min(eve_decrypt_error, decrypt_err)

        return bob_decrypt_error, eve_decrypt_error

    def test_interactive(self):
        def convert(x):
            return np.array([[2*int(i)-1 for i in x]])

        while True:
            P = convert(input('MSG> ')[:self.msg_len])
            K = convert(input('KEY> ')[:self.key_len])

            enc, dec = self.sess.run(
                [self.alice_output, self.bob_output],
                feed_dict={self.msg: P, self.key: K})
            print('ENC>', ''.join(np.where(enc>0, '1', '0')))
            print('DEC>', ''.join(np.where(dec>0, '1', '0')))
