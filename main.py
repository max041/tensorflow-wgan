#!/usr/bin/env python3
import os.path
import pprint
import tensorflow as tf
from data import DATASETS
from model import WGAN
from train import train, train_original
import utils
import random
import numpy as np


flags = tf.app.flags
flags.DEFINE_string('dataset', 'mnist', 'dataset to use {}'.format(
    DATASETS.keys()
))
flags.DEFINE_bool('resize', True, 'whether to resize images on the fly or not')
flags.DEFINE_bool(
    'crop', True,
    'whether to use crop for image resizing or not'
)

flags.DEFINE_integer('z_size', 100, 'size of latent code z [100]')
flags.DEFINE_integer('image_size', 32, 'size of image [32]')
flags.DEFINE_integer('channel_size', 1, 'size of channel [1]')
flags.DEFINE_integer(
    'g_filter_number', 64,
    'number of generator\'s filters at the last transposed conv layer'
)
flags.DEFINE_integer(
    'c_filter_number', 64,
    'number of critic\'s filters at the first conv layer'
)
flags.DEFINE_integer('g_filter_size', 4, 'generator\'s filter size')
flags.DEFINE_integer('c_filter_size', 3, 'discriminator\'s filter size')
flags.DEFINE_float(
    'clip_size', 0.01,
    'parameter clipping size to be applied to the critic'
)

flags.DEFINE_integer('epochs', 10, 'number of the epochs to train')
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_integer('sample_size', 36, 'generator sample size')
flags.DEFINE_float(
    'learning_rate', 0.00002,
    'learning rate for Adam [0.00002]'
)
flags.DEFINE_integer(
    'critic_update_ratio', 2,
    'number of updates for critic parameters per generator\'s updates'
)
flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam [0.5]')
flags.DEFINE_integer(
    'loss_log_interval', 30,
    'number of batches per logging losses'
)
flags.DEFINE_integer(
    'image_log_interval', 100,
    'number of batches per logging sample images'
)
flags.DEFINE_integer(
    'checkpoint_interval', 5000,
    'number of batches per checkpoints'
)

flags.DEFINE_bool('test', False, 'flag defining whether it is in test mode')
flags.DEFINE_boolean('resume', False, 'whether to resume training or not')
flags.DEFINE_string('log_dir', 'logs', 'directory of summary logs')
flags.DEFINE_string('sample_dir', 'figures', 'directory of generated figures')
flags.DEFINE_string(
    'checkpoint_dir', 'checkpoints', 'directory of model checkpoints'
)

# misc
flags.DEFINE_integer('seed', 0, 'Set seed when seed>0.')
flags.DEFINE_string('critic_dump_to', None, 'Dump full step of critic training into specified dir.')
flags.DEFINE_string('generator_dump_to', None, 'Dump full step of generator training into specified dir.')
flags.DEFINE_string('execution_graph_dump_to', None, 'Dump full execution graph into specified dir.')
flags.DEFINE_bool('use_original_algorithm', False, 'Train with algorithm proposed `Wasserstein GAN` paper.')

# Flags quantization manager.
from tf_quantizer.utils import *
flags.DEFINE_bool(name='dump', default=False, help='Dump tensors.')
flags.DEFINE_bool(name='audit', default=False, help='Print detailed information about ops overriding.')
flags.DEFINE_list(name='trunc', default=[],
                  help='Truncate to bfloat inputs to the ops selected from the list:\n' + ' '.join(
                      [s + '_{io}' for s in args_for_trunc]) + '\n Where \'i\' truncates inputs and \'o\' truncates outputs.')
flags.DEFINE_list(name='hw_accurate', default=[],
                  help='HW accurate operations from the list:\n' + ' '.join(args_for_hw))
flags.DEFINE_list(name='special', default=[],
                  help="Ops which aren't hw accurate but do more than only truncation:\n" + ' '.join(
                      args_for_special) + "\nInternaly softmax can use LUTs, but all computations will be done in fp32.")
flags.DEFINE_list(name='disable_softmax_lut', default=[],
                  help='Disable LUTs in softmax which will be used with bfloat implemenation:\n' + ' '.join(
                      args_for_disable_softmax_lut))
flags.DEFINE_bool(name='enable_softmax_ew_trunc', default=False,
                  help="Enable truncation of element-wise operations in softmax.")

FLAGS = flags.FLAGS


def _patch_flags_with_dataset_configuration(flags_):
    flags_.image_size = (
        DATASETS[flags_.dataset].image_size or
        flags_.image_size
    )
    flags_.channel_size = (
        DATASETS[flags_.dataset].channel_size or
        flags_.channel_size
    )
    return flags_


def main(_):
    global FLAGS

    # Set seed if need.
    if FLAGS.seed > 0:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

    # Initialize quantization manager.
    from tf_quantizer.quantization_manager import QuantizationManager as qm
    keys = ['dump', 'audit', 'trunc', 'hw_accurate', 'special', 'disable_softmax_lut', 'enable_softmax_ew_trunc']
    qargs = {}
    for key in keys:
        qargs[key] = FLAGS[key].value
    qm(qargs=qargs).add_quantization()

    # patch and display flags with dataset's width and height
    FLAGS = _patch_flags_with_dataset_configuration(FLAGS)
    pprint.PrettyPrinter().pprint(FLAGS.__flags)

    # compile the model
    wgan = WGAN(
        label=FLAGS.dataset,
        z_size=FLAGS.z_size,
        image_size=FLAGS.image_size,
        channel_size=FLAGS.channel_size,
        g_filter_number=FLAGS.g_filter_number,
        c_filter_number=FLAGS.c_filter_number,
        g_filter_size=FLAGS.g_filter_size,
        c_filter_size=FLAGS.c_filter_size,
    )

    # test / train the model
    if FLAGS.test:
        with tf.Session() as sess:
            name = '{}_test_figures'.format(wgan.name)
            utils.load_checkpoint(sess, wgan, FLAGS)
            utils.test_samples(sess, wgan, name, FLAGS)
            print('=> generated test figures for {} at {}'.format(
                wgan.name, os.path.join(FLAGS.sample_dir, name)
            ))
    elif FLAGS.use_original_algorithm:
        train_original(wgan, FLAGS)
    else:
        train(wgan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
