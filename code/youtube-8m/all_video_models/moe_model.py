import math
import models
import tensorflow as tf
import numpy as np
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS
#
# class MoeModel(models.BaseModel):
#     """A softmax over a mixture of logistic models (with L2 regularization)."""
#
#     def create_model(self,
#                      model_input,
#                      vocab_size,
#                      is_training,
#                      num_mixtures=None,
#                      l2_penalty=1e-8,
#                      **unused_params):
#         """Creates a Mixture of (Logistic) Experts model.
#          It also includes the possibility of gating the probabilities
#
#          The model consists of a per-class softmax distribution over a
#          configurable number of logistic classifiers. One of the classifiers in the
#          mixture is not trained, and always predicts 0.
#
#         Args:
#           model_input: 'batch_size' x 'num_features' matrix of input features.
#           vocab_size: The number of classes in the dataset.
#           is_training: Is this the training phase ?
#           num_mixtures: The number of mixtures (excluding a dummy 'expert' that
#             always predicts the non-existence of an entity).
#           l2_penalty: How much to penalize the squared magnitudes of parameter
#             values.
#         Returns:
#           A dictionary with a tensor containing the probability predictions of the
#           model in the 'predictions' key. The dimensions of the tensor are
#           batch_size x num_classes.
#         """
#         num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
#         low_rank_gating = FLAGS.moe_low_rank_gating
#         l2_penalty = FLAGS.moe_l2;
#         gating_probabilities = FLAGS.moe_prob_gating
#         gating_input = FLAGS.moe_prob_gating_input
#
#         input_size = model_input.get_shape().as_list()[1]
#         remove_diag = FLAGS.gating_remove_diag
#
#         if low_rank_gating == -1:
#             gate_activations = slim.fully_connected(
#                 model_input,
#                 vocab_size * (num_mixtures + 1),
#                 activation_fn=None,
#                 biases_initializer=None,
#                 weights_regularizer=slim.l2_regularizer(l2_penalty),
#                 scope="gates")
#         else:
#             gate_activations1 = slim.fully_connected(
#                 model_input,
#                 low_rank_gating,
#                 activation_fn=None,
#                 biases_initializer=None,
#                 weights_regularizer=slim.l2_regularizer(l2_penalty),
#                 scope="gates1")
#             gate_activations = slim.fully_connected(
#                 gate_activations1,
#                 vocab_size * (num_mixtures + 1),
#                 activation_fn=None,
#                 biases_initializer=None,
#                 weights_regularizer=slim.l2_regularizer(l2_penalty),
#                 scope="gates2")
#
#         expert_activations = slim.fully_connected(
#             model_input,
#             vocab_size * num_mixtures,
#             activation_fn=None,
#             weights_regularizer=slim.l2_regularizer(l2_penalty),
#             scope="experts")
#
#         gating_distribution = tf.nn.softmax(tf.reshape(
#             gate_activations,
#             [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
#         expert_distribution = tf.nn.sigmoid(tf.reshape(
#             expert_activations,
#             [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures
#
#         probabilities_by_class_and_batch = tf.reduce_sum(
#             gating_distribution[:, :num_mixtures] * expert_distribution, 1)
#         probabilities = tf.reshape(probabilities_by_class_and_batch,
#                                    [-1, vocab_size])
#
#         if gating_probabilities:
#             if gating_input == 'prob':
#                 gating_weights = tf.get_variable("gating_prob_weights",
#                                                  [vocab_size, vocab_size],
#                                                  initializer=tf.random_normal_initializer(
#                                                      stddev=1 / math.sqrt(vocab_size)))
#                 gates = tf.matmul(probabilities, gating_weights)
#             else:
#                 gating_weights = tf.get_variable("gating_prob_weights",
#                                                  [input_size, vocab_size],
#                                                  initializer=tf.random_normal_initializer(
#                                                      stddev=1 / math.sqrt(vocab_size)))
#
#                 gates = tf.matmul(model_input, gating_weights)
#
#             if remove_diag:
#                 # removes diagonals coefficients
#                 diagonals = tf.matrix_diag_part(gating_weights)
#                 gates = gates - tf.multiply(diagonals, probabilities)
#
#             gates = slim.batch_norm(
#                 gates,
#                 center=True,
#                 scale=True,
#                 is_training=is_training,
#                 scope="gating_prob_bn")
#
#             gates = tf.sigmoid(gates)
#
#             probabilities = tf.multiply(probabilities, gates)
#
#         return {"predictions": probabilities}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.
     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.
    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}