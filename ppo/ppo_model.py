import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

class PPOModel:

    def __init__(self):
        self.normalize = 0

    def _create_continuous_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders.

        activation:
            What type of activation function to use for layers.
        h_size:
            Hidden layer size.
        num_streams:
            Number of state streams to construct.
        s_size:
            state input size.

        return:
            List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='state')
        if self.normalize > 0:
            self.running_mean = tf.get_variable('running_mean',
                                                [s_size],
                                                trainable=False,
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
            self.running_variance = tf.get_variable('running_variance',
                                                    [s_size],
                                                    trainable=False,
                                                    dtype=tf.float32,
                                                    initializer=tf.ones_initializer())
            self.norm_running_variance = tf.get_variable('norm_running_variance',
                                                         [s_size],
                                                         trainable=False,
                                                         dtype=tf.float32,
                                                         initializer=tf.ones_initializer())
            self.normalized_state = tf.clip_by_value(
                (self.state_in - self.running_mean) / tf.sqrt(self.norm_running_variance), -5, 5, name='normalized_state')
            self.new_mean = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_mean')
            self.new_variance = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_variance')
            self.update_mean = tf.assign(self.running_mean, self.new_mean)
            self.update_variance = tf.assign(self.running_variance, self.new_variance)
            self.update_norm_variance = tf.assign(self.norm_running_variance,
                                                  self.running_variance / (tf.cast(self.global_step, tf.float32) + 1))
        else:
            self.normalized_state = self.state_in
        streams = []
        for _ in range(num_streams):
            hidden = self.normalized_state
            for _ in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def _create_discrete_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders from discrete state input.

        s_size:
            state input size (discrete).
        h_size:
            Hidden layer size.
        num_streams:
            Number of state streams to construct.
        activation:
            What type of activation function to use for layers.

        return:
            List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='state')
        state_in = tf.reshape(self.state_in, [-1])
        state_onehot = c_layers.one_hot_encoding(state_in, s_size)
        streams = []
        hidden = state_onehot
        for _ in range(num_streams):
            for _ in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def _create_global_steps(self):
        """
        Creates TF ops to track and increment global training step.
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.increment_step = tf.assign(self.global_step, tf.cast(self.global_step, tf.int32) + 1)

    def _create_ppo_optimizer(self, probs, old_probs, value, entropy, beta, epsilon, lr, max_step):
        """
        Creates training-specific Tensorflow ops for PPO models.

        beta:
            Entropy regularization strength
        entropy:
            Current policy entropy
        epsilon:
            Value for policy-divergence threshold
        lr:
            Learning rate
        max_step:
            Total number of training steps.
        old_probs:
            Past policy probabilities
        probs:
            Current policy probabilities
        value:
            Current value estimate
        """
        self.returns_holder = tf.placeholder(tf.float32, shape=[None], name='discounted_rewards')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
        decay_epsilon = tf.train.polynomial_decay(epsilon,
                                                  self.global_step,
                                                  decay_steps=max_step,
                                                  end_learning_rate=1e-2)
        r_theta = probs / (old_probs + 1e-10)
        p_opt_a = r_theta * self.advantage
        p_opt_b = tf.clip_by_value(r_theta, 1 - decay_epsilon, 1 + decay_epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))    # pylint:disable=E1130
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.returns_holder, tf.reduce_sum(value, axis=1)))
        decay_beta = tf.train.polynomial_decay(beta, self.global_step, max_step, 1e-5, power=1.0)
        self.loss = self.policy_loss + self.value_loss - decay_beta * tf.reduce_mean(entropy)
        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step, max_step, 1e-10, power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

    def _create_reward_encoder(self):
        """
        Creates TF ops to track and increment recent average cumulative reward.
        """
        self.last_reward = tf.Variable(0, name='last_reward', trainable=False, dtype=tf.float32)
        self.new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        self.update_reward = tf.assign(self.last_reward, self.new_reward)

    def _create_visual_encoder(self, o_size_h, o_size_w, bw, h_size, num_streams, activation, num_layers):
        """
        Builds a set of visual (CNN) encoders.

        activation:
            What type of activation function to use for layers.
        bw:
            Whether image is greyscale {True} or color {False}.
        h_size:
            Hidden layer size.
        num_streams:
            Number of visual streams to construct.
        o_size_h:
            Height observation size.
        o_size_w:
            Width observation size.

        return:
            List of hidden layer tensors.
        """
        if bw:
            c_channels = 1
        else:
            c_channels = 3
        self.observation_in = tf.placeholder(tf.float32,
                                             shape=[None, o_size_h, o_size_w, c_channels],
                                             name='observation_0')
        streams = []
        for _ in range(num_streams):
            self.conv1 = tf.layers.conv2d(self.observation_in,
                                          activation=activation,
                                          filters=16,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          use_bias=False)
            self.conv2 = tf.layers.conv2d(self.conv1,
                                          activation=activation,
                                          filters=32,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          use_bias=False)
            hidden = c_layers.flatten(self.conv2)
            for _ in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams
