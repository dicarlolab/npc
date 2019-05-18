import tensorflow as tf
from functools import reduce


class Loss(object):
  def __init__(self, model, weight=1.):
    self._model = model
    self._weight = weight

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class CustomScore(Loss):
  def __init__(self, model, weight, score_func):
    super(CustomScore, self).__init__(model, weight)
    self._score_func = score_func

  def __call__(self):
    return -self._weight * self._score_func


class TvLoss(Loss):
  def __init__(self, model, weight):
    super(TvLoss, self).__init__(model, weight)

  @staticmethod
  def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

  def __call__(self):  # *args, **kwargs
    image = self._model.image_ph
    shape = [d.value for d in image.get_shape()]

    tv_y_size = self._tensor_size(image[:, 1:, :, :])
    tv_x_size = self._tensor_size(image[:, :, 1:, :])
    return self._weight * (
      (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
       tv_y_size) +
      (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
       tv_x_size)
    )
