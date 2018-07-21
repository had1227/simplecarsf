from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

import numpy as np
import tensorflow as tf


def spmax(logits):

    z_sorted = np.sort(logits)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    k = np.arange(1,len(z_sorted)+1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = np.sum(z_check)
    tau_sum = np.sum(z_sorted * z_check)
    tau = (tau_sum-1) / k_z

    z_spmax = 0.5 + 0.5 * np.sum(z_sorted[:k_z]**2-tau**2)
    
    return z_spmax



__all__ = ["sparsemax"]


def tf_spmax(logits, name=None):
  """Computes sparsemax activations [1].
  For each batch `i` and class `j` we have
    $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
  [1]: https://arxiv.org/abs/1602.02068
  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `logits`.
  """

  with ops.name_scope(name, "sparsemax", [logits]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    obs = array_ops.shape(logits)[0]
    dims = array_ops.shape(logits)[1]

    z = logits #- math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

    # sort z
    z_sorted, _ = nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = math_ops.cumsum(z_sorted, axis=1)
    k = math_ops.range(
        1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

    # calculate tau(z)
    indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
    tau_sum = array_ops.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

    spmax_policy = math_ops.maximum(
        math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])

    z_square = math_ops.square(z_sorted)
    tau_square = math_ops.square(tau_z)
    spmax = 0.5 * (math_ops.reduce_sum(math_ops.cast(z_check, dtypes.float32) * z_square, axis=1) - math_ops.cast(k_z,dtypes.float32) * tau_square) + 0.5

    return spmax_policy, spmax

    # calculate p
    #return math_ops.maximum(
    #    math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])

'''
print("tf")
logits=np.asarray([[0,1,3.5,4]])
ph = tf.placeholder(tf.float32,[1,4])

pol,sp,zs,ts,tz = tf_spmax(ph)

sess= tf.Session()

result = sess.run([pol,sp,zs,ts,tz],feed_dict={ph:logits})

print (result)
'''
