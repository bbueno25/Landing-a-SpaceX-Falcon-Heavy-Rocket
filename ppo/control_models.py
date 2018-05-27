import numpy as np
import os
import ppo_model
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

class ContinuousControlModel(ppo_model.PPOModel):
    """
    Creates Continuous Control Actor-Critic model.

    brain:
        State-space size
    h_size:
        Hidden layer size
    """
    def __init__(self, brain, epsilon, h_size, lr, max_step, normalize, num_layers):
        super(ContinuousControlModel, self).__init__()
        s_size = brain.state_space_size
        a_size = brain.action_space_size
        self.normalize = normalize
        self._create_global_steps()
        self._create_reward_encoder()
        hidden_state, hidden_visual, hidden_policy, hidden_value = None, None, None, None
        if brain.number_observations > 0:
            height_size, width_size = brain.camera_resolutions[0]['height'], brain.camera_resolutions[0]['width']
            bw = brain.camera_resolutions[0]['blackAndWhite']
            hidden_visual = self._create_visual_encoder(height_size, width_size, bw, h_size, 2, tf.nn.tanh, num_layers)
        if brain.state_space_size > 0:
            s_size = brain.state_space_size
            if brain.state_space_type == 'continuous':
                hidden_state = self._create_continuous_state_encoder(s_size, h_size, 2, tf.nn.tanh, num_layers)
            else:
                hidden_state = self._create_discrete_state_encoder(s_size, h_size, 2, tf.nn.tanh, num_layers)
        if hidden_visual is None and hidden_state is None:
            raise Exception('No valid network configuration possible. There are no states or observations in this brain.')
        elif hidden_visual is not None and hidden_state is None:
            hidden_policy, hidden_value = hidden_visual
        elif hidden_visual is None and hidden_state is not None:
            hidden_policy, hidden_value = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden_policy = tf.concat([hidden_visual[0], hidden_state[0]], axis=1)
            hidden_value = tf.concat([hidden_visual[1], hidden_state[1]], axis=1)
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.mu = tf.layers.dense(hidden_policy,
                                  a_size,
                                  activation=None,
                                  use_bias=False,
                                  kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))
        self.log_sigma_sq = tf.get_variable('log_sigma_squared',
                                            [a_size],
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
        self.sigma_sq = tf.exp(self.log_sigma_sq)
        self.epsilon = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='epsilon')
        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output_max = tf.identity(self.mu, name='action_max')
        self.output = tf.identity(self.output, name='action')
        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.probs = tf.multiply(a, b, name='action_probs')
        self.entropy = tf.reduce_sum(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))
        self.value = tf.layers.dense(hidden_value, 1, activation=None, use_bias=False)
        self.value = tf.identity(self.value, name='value_estimate')
        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')
        self._create_ppo_optimizer(self.probs, self.old_probs, self.value, self.entropy, 0.0, epsilon, lr, max_step)

class DiscreteControlModel(ppo_model.PPOModel):
    """
    Creates Discrete Control Actor-Critic model.

    brain:
        State-space size
    h_size:
        Hidden layer size
    """
    def __init__(self, beta, brain, epsilon, h_size, lr, max_step, normalize, num_layers):
        super(DiscreteControlModel, self).__init__()
        self._create_global_steps()
        self._create_reward_encoder()
        self.normalize = normalize
        hidden_state, hidden_visual, hidden = None, None, None
        if brain.number_observations > 0:
            height_size, width_size = brain.camera_resolutions[0]['height'], brain.camera_resolutions[0]['width']
            bw = brain.camera_resolutions[0]['blackAndWhite']
            hidden_visual = self._create_visual_encoder(height_size, width_size, bw, h_size, 1, tf.nn.elu, num_layers)[0]
        if brain.state_space_size > 0:
            s_size = brain.state_space_size
            if brain.state_space_type == 'continuous':
                hidden_state = self._create_continuous_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]
            else:
                hidden_state = self._create_discrete_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]
        if hidden_visual is None and hidden_state is None:
            raise Exception('No valid network configuration possible. There are no states or observations in this brain.')
        elif hidden_visual is not None and hidden_state is None:
            hidden = hidden_visual
        elif hidden_visual is None and hidden_state is not None:
            hidden = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden = tf.concat([hidden_visual, hidden_state], axis=1)
        a_size = brain.action_space_size
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.policy = tf.layers.dense(hidden,
                                      a_size,
                                      activation=None,
                                      use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))
        self.probs = tf.nn.softmax(self.policy, name='action_probs')
        self.output = tf.multinomial(self.policy, 1)
        self.output_max = tf.argmax(self.probs, name='action_max', axis=1)
        self.output = tf.identity(self.output, name='action')
        self.value = tf.layers.dense(hidden, 1,
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=c_layers.variance_scaling_initializer(factor=1.0))
        self.value = tf.identity(self.value, name='value_estimate')
        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs + 1e-10), axis=1)    # pylint:disable=E1130
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, a_size)
        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')
        self.responsible_probs = tf.reduce_sum(self.probs * self.selected_actions, axis=1)
        self.old_responsible_probs = tf.reduce_sum(self.old_probs * self.selected_actions, axis=1)
        self._create_ppo_optimizer(self.responsible_probs,
                                   self.old_responsible_probs,
                                   self.value,
                                   self.entropy,
                                   beta,
                                   epsilon,
                                   lr,
                                   max_step)
