import control_models
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow.python.tools.freeze_graph as freeze_graph

def create_agent_model(env,
                       beta=1e-3,
                       epsilon=0.2,
                       h_size=128,
                       lr=1e-4,
                       max_step=5e6,
                       normalize=0,
                       num_layers=2):
    """
    Takes a OpenAI Gym environment and model-specific hyper-parameters
    and returns the appropriate PPO agent model for the environment.

    beta:
        Strength of entropy regularization.
    env:
        an OpenAI Gym environment.
    epsilon:
        Value for policy-divergence threshold.
    h_size:
        Size of hidden layers.
    lr:
        Learning rate.
    max_step:
        Total number of training steps.

    return:
        a sub-class of PPOAgent tailored to the environment.
    """
    if num_layers < 1:
        num_layers = 1
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    if brain.action_space_type == 'discrete':
        return control_models.DiscreteControlModel(beta,
                                                   brain,
                                                   epsilon,
                                                   h_size,
                                                   lr,
                                                   max_step,
                                                   normalize,
                                                   num_layers)
    return control_models.ContinuousControlModel(brain,
                                                 epsilon,
                                                 h_size,
                                                 lr,
                                                 max_step,
                                                 normalize,
                                                 num_layers)

def export_graph(model_path, env_name='env', target_nodes='action, value_estimate, action_probs'):
    """
    Exports latest saved model to .bytes format for Unity embedding.

    env_name:
        Name of associated Learning Environment.
    model_path:
        path of model checkpoints.
    target_nodes:
        Comma separated string of needed output nodes for
        embedded graph.
    """
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=os.path.join(model_path, 'graph.pbtxt'),
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,    # pylint:disable=E1101
                              output_node_names=target_nodes,
                              output_graph=os.path.join(model_path, env_name, '.bytes'),
                              clear_devices=True,
                              initializer_nodes='',
                              input_saver='',
                              restore_op_name='save\\restore_all',
                              filename_tensor_name='save\\Const:0')

def save_model(sess, saver, model_path='.\\', steps=0):
    """
    Saves current model to checkpoint folder.

    model_path:
        Designated model path.
    saver:
        Tensorflow saver for session.
    sess:
        Current Tensorflow session.
    steps:
        Current number of steps in training process.
    """
    last_checkpoint = os.path.join(model_path, 'model', '.cptk-', str(steps))
    saver.save(sess, last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'graph.pbtxt', as_text=False)
    logging.info('model:saved')
