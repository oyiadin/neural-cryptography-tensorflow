

################################
# original author: @ankeshanand
# repo url: https://github.com/ankeshanand/neural-cryptography-tensorflow
# 
# slightly modified by @oyiadin
################################


import os
import string
import random
import logging
import binascii
import numpy as np
import tensorflow as tf

from model import CryptoNet

import tornado.ioloop
import tornado.web
from tornado.options import define, options, parse_command_line

define("port", default=13577, help="run on the given port", type=int)
define("debug", default=False, help="run in debug mode")

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

LEN = 96
KEY = 'D33PLeARn!nG'
KEY = list(map(int, bin(int(binascii.hexlify(KEY.encode()), base=16))[2:].zfill(LEN)))


class BaseHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write(dict(err_msg='unsupported'))

    def post(self, *args, **kwargs):
        self.write(dict(err_msg='unsupported'))


class IndexHandler(BaseHandler):
    def get(self):
        logger.info('[idx] {}'.format(self.request.remote_ip))
        self.render("index.html")


class CryptoHandler(BaseHandler):
    def get(self):
        msg = self.get_query_argument('msg', default=None)
        key = self.get_query_argument('key', default=None)
        logger.info('[enc] {} msg={}, key={}'.format(
            self.request.remote_ip, msg, key))

        if not msg:
            self.write(dict(err_msg='argument `msg` is required'))
            return

        try:
            msg = list(map(float, msg.rstrip(',').split(',')))
            if not key:
                key = KEY
            else:
                key = list(map(float, key.rstrip(',').split(',')))
        except ValueError:
            self.write(dict(
                err_msg="illegal msg/key: should be something like: 0.0,1.0,-1.0,... (no space, only float and `,`)"))
            return

        try:
            cipher = crypto_net.encrypt_hex(msg, key)
        except AssertionError as e:
            self.write(dict(err_msg=str(e)))
            return

        raw_cipher = ','.join(map(str, cipher))
        cipher = ''.join(map(str, np.where(cipher>0, 1, 0)))
        self.write(dict(
            raw_cipher=raw_cipher,
            cipher_bin=cipher,
            cipher_hex=hex(int(cipher, base=2)),
            err_msg=''))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/enc", CryptoHandler),
    ], template_path="templates")


if __name__ == '__main__':
    logger.info('initilizing the tensorflow session and graph...')
    sess = tf.Session()
    logger.info('loading model')
    crypto_net = CryptoNet(
        sess=sess, msg_len=96,
        model_path=os.path.join('saved-model', 'alice_bob'))

    # crypto_net.test_interactive()

    app = make_app()
    app.listen(options.port)
    logger.info('starting listening...')
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info('exiting')
