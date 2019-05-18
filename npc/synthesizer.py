from __future__ import print_function

import tensorflow as tf
import numpy as np
import copy

npa = np.array


class Synthesizer(object):
  def __init__(self, model, losses):
    self._graph = model.graph
    self._sess = model.sess
    self._image = model.image_ph
    self._losses = losses
    self._model = model

  @staticmethod
  def _jitter_image(img, max_pixels=19):
    sx, sy = np.random.randint(-max_pixels, max_pixels, size=2)
    img_shift = np.roll(np.roll(img, sx, axis=1), sy, axis=0)
    return img_shift, sx, sy

  @staticmethod
  def _unjitter_image(img, sx, sy):
    return np.roll(np.roll(img, -sx, axis=1), -sy, axis=0)

  @staticmethod
  def _postprocess(image):
    image = copy.copy(image) * 128.
    image = (image * 0.99 + 128.).astype(np.uint8)
    return image

  def run(self, num_iterations, initial_image, jitter=False, learning_rate=0.001,
          chkpt_period=100, monitor_varaible=None):
    with self._graph.as_default():
      run_feed_dict = {self._image: initial_image}
      img = initial_image.copy()
      with tf.variable_scope('loss'):
        total_loss = tf.reduce_sum([loss() for loss in self._losses])
        grads = tf.gradients(total_loss, self._image)[0]
        g_bias, g_var = tf.nn.moments(tf.reshape(grads, [-1]), axes=[0])
        grads /= tf.sqrt(g_var) + 1e-8

      init_loss = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss'))
      self._sess.run(init_loss)
      outputs, features_cache, chkpoints = [], [], []
      for i in range(num_iterations):
        if jitter:
          sub, sx, sy = self._jitter_image(img)
          run_feed_dict.update({self._image: sub})
          gs = self._sess.run(grads, feed_dict=run_feed_dict)
          gs = self._unjitter_image(gs, sx, sy)
        else:
          gs = self._sess.run(grads, feed_dict=run_feed_dict)
        img = np.clip(img - gs * learning_rate, -1, 1)
        run_feed_dict.update({self._image: img})

        if (i % chkpt_period == 0) or (i == num_iterations - 1):
          print('Scores: {0}, Saving checkpoint - {1}'.format(total_loss.eval(feed_dict=run_feed_dict), i))
          features_cache.append(self._losses[0]().eval(feed_dict=run_feed_dict))
          chkpoints.append(self._postprocess(img))
          if monitor_varaible is not None:
            outputs.append(self._model.endpoints[monitor_varaible].eval(feed_dict=run_feed_dict))

      return features_cache, chkpoints, outputs


def main():
  import h5py
  import tensorflow as tf
  import numpy as np
  from npc.model import Model
  # import npc.synthesizer as synthesizer
  import npc.losses as losses
  import matplotlib.pyplot as plt

  npa = np.array

  neuron_id = 45
  mode = 'ohp'  # ['stretch', 'ohp']
  assert mode in ['stretch', 'ohp']

  with h5py.File(
    '/braintree/data2/active/users/bashivan/results/dimensionality/synth_images/v4_sep_weights_magneto_retina_season9.h5') \
    as h5file:
    weights = {k: npa(h5file[k]) for k in h5file.keys()}

  checkpoint_path = '/braintree/data2/active/users/bashivan/checkpoints/imagenet_alexnet_raw_/'
  readout_layer = 'conv3'
  output_layer = 'output'
  ph_shape = (1, 299, 299, 1)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

  with tf.Graph().as_default() as g, tf.Session() as sess:
    np.random.seed(0)
    initial_image = np.random.uniform(low=-1, high=1, size=ph_shape)
    initial_image /= 2 * np.max(np.abs(initial_image))

    image_ph = tf.placeholder(tf.float32, shape=ph_shape, name='image_ph')
    model = Model(checkpoint_path=checkpoint_path, model_name='alexnet', image=image_ph,
                  graph=g, sess=sess, gpu_options=gpu_options)
    model.map_output_sep(weights=weights, readout_layer=readout_layer)

    # Construct the score function using TF functions
    if mode == 'stretch':
      score_func = tf.reshape(model.endpoints[output_layer], (-1,))[neuron_id]
    else:
      score_func = tf.reshape(tf.nn.softmax(model.endpoints[output_layer], axis=-1), (-1,))[neuron_id]

    ls = [losses.CustomScore(model, weight=1.0, score_func=score_func),
          losses.TvLoss(model, weight=500.0)]
    synth = Synthesizer(model=model, losses=ls)
    preds, checkpoints, outputs = synth.run(num_iterations=200, initial_image=initial_image,
                                            jitter=True, monitor_varaible=output_layer, learning_rate=0.01)
    print(np.squeeze(outputs)[:, neuron_id])
    plt.figure()
    plt.imshow(np.squeeze(checkpoints[-1]), cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
  main()
