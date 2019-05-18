from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import sys
from absl import flags
import numpy as np


npa = np.array
DEFAULT_IMAGE_SHAPE = (1, 299, 299, 1)


class Model(object):
  def __init__(self, checkpoint_path, model_name, image=None, num_classes=1001, graph=None, sess=None, gpu_options=None,
               zoo_path='/braintree/home/bashivan/dropbox/Codes/base_model/Pipeline/Model_zoo/'):
    if graph is None:
      self._graph = tf.Graph()
    else:
      self._graph = graph
    self._checkpoint_path = checkpoint_path
    self._zoo_path = zoo_path

    with self._graph.as_default():
      if image is None:
        self._image_placeholder = tf.placeholder(tf.float32, shape=DEFAULT_IMAGE_SHAPE, name='image_ph')
      else:
        self._image_placeholder = image
      self._image = tf.tile(self._image_placeholder, [1, 1, 1, 3])
      self._image_shape = [i.value for i in self._image_placeholder.shape]

      self._endpoints = self._make_model(images=self._image, model_name=model_name, num_classes=num_classes)
      if sess is None:
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
      else:
        self._sess = sess
      self._initialize_all_vars()
      self._load_checkpoint()

  def _initialize_all_vars(self):
    with self._graph.as_default():
      init_all = tf.global_variables_initializer()
      self._sess.run(init_all)

  def _load_checkpoint(self):
    variables_to_restore = slim.get_variables_to_restore(exclude=['input', 'pca', 'pls', 'loss', 'sep'])
    saver = tf.train.Saver(variables_to_restore)
    if os.path.isdir(self._checkpoint_path):
      ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(self._sess, os.path.join(self._checkpoint_path,
                                                 ckpt.model_checkpoint_path))

        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')
        return
    else:
      saver.restore(self._sess, self._checkpoint_path)

  def inference(self, images, readout_layers):
    endpoints = self._sess.run(self._endpoints, feed_dict={self._image_placeholder: images})
    if type(readout_layers) is list:
      assert all([s in endpoints for s in readout_layers]), \
        'Layer not found. Available layers: {0}'.format(endpoints.keys())
      return {s: endpoints[s] for s in readout_layers}
    else:
      assert (readout_layers in endpoints), 'Layer not found. Available layers: {0}'.format(endpoints.keys())
      return {readout_layers: endpoints[readout_layers]}

  def _make_model(self, images, model_name, num_classes=1001, is_training=False):
    sys.path.insert(0, self._checkpoint_path)
    sys.path.insert(0, self._zoo_path)
    with self._graph.as_default():
      exec('import {0}_model as model_class'.format(model_name))
      model = eval('model_class.{0}()'.format(model_name.title()))
      print('Model loaded.')

      with slim.arg_scope(model.arg_scope()):
        _, endpoints = model.model(
          images,
          num_classes=num_classes,
          is_training=is_training,
          scope=None)
      return endpoints

  def map_output_sep(self, weights, readout_layer):
    with self._graph.as_default():
      assert hasattr(self, '_endpoints')
      num_neurons = weights['s_w'].shape[0]
      with tf.variable_scope('sep'):
        out_layer = self._endpoints[readout_layer]
        preds = []
        for n in range(num_neurons):
          with tf.variable_scope('N_{}'.format(n)):
            s_w = tf.Variable(
              initial_value=weights['s_w'][n].reshape(1, out_layer.shape[1], out_layer.shape[2], 1),
              dtype=tf.float32)
            net = s_w * out_layer
            d_w = tf.Variable(
              initial_value=weights['d_w'][n].reshape(1, 1, net.shape[-1], 1),
              dtype=tf.float32)
            out = tf.nn.conv2d(net, d_w, [1, 1, 1, 1], 'SAME')
            bias = tf.Variable(initial_value=weights['bias'][n], dtype=tf.float32)
            pred = tf.reduce_sum(out, axis=[1, 2]) + bias
            if 'hvm_neural_std' in weights:
              pred *= tf.constant(weights['hvm_neural_std'][n], dtype=tf.float32)
              pred += tf.constant(weights['hvm_neural_mean'][n], dtype=tf.float32)
            preds.append(pred)
        self._endpoints['output'] = tf.concat(preds, axis=1)
        init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sep'))
        self._sess.run(init_op)

  def map_output_reg(self, weights, readout_layer):
    with self._graph.as_default():
      assert hasattr(self, '_endpoints')
      with tf.variable_scope('pca'):
        net = self._endpoints[readout_layer]
        net = slim.flatten(net, scope='flatten')
        net -= weights['pca_bias']
        net = slim.fully_connected(net,
                                   weights['pca_w'].shape[0],
                                   activation_fn=None,
                                   weights_initializer=tf.constant_initializer(weights['pca_w'].T,
                                                                               verify_shape=True),
                                   trainable=False,
                                   scope='pca')
        self._endpoints['pca_out'] = net
      init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pca'))
      self._sess.run(init_op)
      with tf.variable_scope('reg'):
        net = slim.fully_connected(net,
                                   weights['reg_w'].shape[0],
                                   activation_fn=None,
                                   weights_initializer=tf.constant_initializer(weights['reg_w'].T,
                                                                               verify_shape=True),
                                   trainable=False,
                                   scope='reg')
        net += weights['reg_bias']
        self._endpoints['output'] = net
      init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reg'))
      self._sess.run(init_op)

  @property
  def graph(self):
    return self._graph

  @property
  def sess(self):
    return self._sess

  @property
  def image_ph(self):
    return self._image_placeholder

  @property
  def endpoints(self):
    return self._endpoints


def preprocess(im):
  return (npa(im, dtype=np.float) - 128.) / 128.


