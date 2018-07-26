import math

from .. import models
from .. import video_level_models
import tensorflow as tf
from .. import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS

class GruModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
        """Creates a model which uses a stack of GRUs to represent the video.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        gru_size = FLAGS.gru_cells
        number_of_layers = FLAGS.gru_layers
        backward = FLAGS.gru_backward
        random_frames = FLAGS.gru_random_sequence
        iterations = FLAGS.iterations

        if random_frames:
            num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
            model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                                   iterations)

        if backward:
            model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1)

        stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
            ], state_is_tuple=False)

        loss = 0.0
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                               sequence_length=num_frames,
                                               dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)