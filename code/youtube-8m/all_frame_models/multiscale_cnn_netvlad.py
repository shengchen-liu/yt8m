import math
import models
import tensorflow as tf
import numpy as np
# import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

# import video_level_models
import model_utils as utils

class NetVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keepdims =True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad

class MultiscaleCnnNetVLAD(models.BaseModel):
    """Creates a NetVLAD based model.

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


    def moe(self, model_input,
            vocab_size,
            num_mixtures=None,
            l2_penalty=1e-8,
            scopename="",
            **unused_params):

        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates" + scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts" + scopename)

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

        return final_probabilities

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[1024, 1024, 1024],
            filter_sizes=[1, 2, 3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in xrange(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0, 0], [i, 0], [0, 0]])[:, :max_frames, :])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope + "cnn-filter-len%d" % fs,
                                         shape=[num_features * fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.is_training,
            scope=sub_scope + "cluster_bn")
        return cnn_output

    def netvlad_layer(self,
                      model_input,
                      num_frames,
                      add_batch_norm=None,
                      sample_random_frames=None,
                      cluster_size=None,
                      hidden_size=None,
                      is_training=True,
                      sub_scope="",
                      **unused_params):

        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.netvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
        relu = FLAGS.netvlad_relu
        dimred = FLAGS.netvlad_dimred
        gating = FLAGS.gating
        remove_diag = FLAGS.gating_remove_diag
        lightvlad = FLAGS.lightvlad
        vlagd = FLAGS.vlagd

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])



        video_NetVLAD = NetVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
        # audio_NetVLAD = NetVLAD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

        if add_batch_norm:  # and not lightvlad:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope=sub_scope + "input_bn")

        with tf.variable_scope(sub_scope + "video_VLAD"):
            vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024])

        # with tf.variable_scope(sub_scope + "audio_VLAD"):
        #     vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])
        vlad = vlad_video
        # vlad = tf.concat([vlad_video, vlad_audio], 1)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable(sub_scope + "hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(vlad, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope=sub_scope + "hidden1_bn")

        else:
            hidden1_biases = tf.get_variable(sub_scope + "hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable(sub_scope + "gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))

            gates = tf.matmul(activation, gating_weights)

            if remove_diag:
                # removes diagonals coefficients
                diagonals = tf.matrix_diag_part(gating_weights)
                gates = gates - tf.multiply(diagonals, activation)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope=sub_scope + "gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)
        return activation

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):

        num_layers = FLAGS.multiscale_cnn_lstm_layers
        pool_size = 2
        num_filters = [256, 256, 512]
        filter_sizes = [1, 2, 3]
        features_size = sum(num_filters)

        sub_predictions = []
        cnn_input = model_input
        cnn_max_frames = model_input.get_shape().as_list()[1]
        with tf.variable_scope("Multiscale_CNN"):
            for layer in range(num_layers):
                cnn_output = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes,
                                      sub_scope="cnn%d" % (layer + 1))
                cnn_output_relu = tf.nn.relu(cnn_output)

                netvlad_memmory = self.netvlad_layer(cnn_output_relu, num_frames=num_frames, is_training=True, sub_scope="netvlad%d" % (layer + 1))


                # lstm_memory = self.rnn(cnn_output_relu, lstm_size, num_frames, sub_scope="rnn%d" % (layer + 1))
                sub_prediction = self.moe(netvlad_memmory, vocab_size, scopename="moe%d"%(layer+1))
                sub_predictions.append(sub_prediction)
                cnn_max_frames /= pool_size
                max_pooled_cnn_output = tf.reduce_max(
                    tf.reshape(
                        cnn_output_relu[:, :cnn_max_frames * 2, :],
                        [-1, cnn_max_frames, pool_size, features_size]
                    ), axis=2)

                # for the next cnn layer
                cnn_input = max_pooled_cnn_output
                num_frames = tf.maximum(num_frames / pool_size, 1)

            support_predictions = tf.concat(sub_predictions, axis=1)

            # classifer consensus
            predictions = tf.add_n(sub_predictions) / len(sub_predictions)

            return {"predictions": predictions,
                    "support_predictions": support_predictions}





