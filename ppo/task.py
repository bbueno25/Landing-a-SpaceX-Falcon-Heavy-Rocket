"""

Proximal Policy Optimization (PPO)

Contains an implementation of PPO as described here: https://arxiv.org/abs/1707.06347

"""
import argparse
import agent
import environment as environ
import logging
import models
import numpy as np
import os
import renderthread
import shutil
import tensorflow as tf
import time
import trainer as tnr

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size',
                        default=256,
                        help='How many experiences per gradient descent update step [default: 64].')
    parser.add_argument('--beta',
                        default=2.5e-3,
                        help='Strength of entropy regularization [default: 2.5e-3].')
    parser.add_argument('--buffer-size',
                        default=2.5e-3 * 16,
                        help='How large the experience buffer should be before gradient descent [default: 2048].')
    parser.add_argument('--env_name',
                        default='RocketLander-v0',
                        help='Name of environment.')
    parser.add_argument('--epsilon',
                        default=0.2,
                        help='Acceptable threshold around ratio of old and new policy probabilities [default: 0.2].')
    parser.add_argument('--gamma',
                        default=0.99,
                        help='Reward discount rate [default: 0.99].')
    parser.add_argument('--hidden_units',
                        default=128,
                        help='Number of units in hidden layer [default: 64].')
    parser.add_argument('--keep_checkpoints',
                        default=5,
                        help='How many model checkpoints to keep [default: 5].')
    parser.add_argument('--lambd',
                        default=0.95,
                        help='Lambda parameter for GAE [default: 0.95].')
    parser.add_argument('--learning_rats',
                        default=1e-4,
                        help='Model learning rate [default: 3e-4].')
    parser.add_argument('--load_model',
                        default=True,
                        help='Whether to load the model or randomly initialize [default: False].')
    parser.add_argument('--max_steps',
                        default=20e6,
                        help='Maximum number of steps to run environment [default: 1e6].')
    parser.add_argument('--model_path',
                        default='./models',
                        help='The sub-directory name for model and summary statistics.')
    parser.add_argument('--normalize_steps',
                        default=10e6,
                        help='Activate state normalization for this many steps and freeze statistics afterwards.')
    parser.add_argument('--num_epoch',
                        default=10,
                        help='Number of gradient descent steps per batch of experiences [default: 5].')
    parser.add_argument('--num_layers',
                        default=1,
                        help='Number of hidden layers between state/observation and outputs [default: 2].')
    parser.add_argument('--record',
                        default=False,
                        help='Save recordings of episodes.')
    parser.add_argument('--render',
                        default=True,
                        help='Render environment to display progress.')
    parser.add_argument('--save_freq',
                        default=2.5e-3 * 5,
                        help='Frequency at which to save training statistics [default: 50000].')
    parser.add_argument('--summary_freq',
                        default=2.5e-3 * 5,
                        help='Frequency at which to save training statistics [default: 10000].')
    parser.add_argument('--summary_path',
                        default='./ppo_summary',
                        help='The sub-directory name for model and summary statistics.')
    parser.add_argument('--time_horizon',
                        default=2048,
                        help='How many steps to collect per agent before adding to buffer [default: 2048].')
    parser.add_argument('--train_model',
                        default=True,
                        help='Whether to train model, or only run inference [default: False].')
    args = parser.parse_args()
    env = environ.GymEnvironment(args.env_name, log_path='./ppo_log', skip_frames=5)
    env_render = environ.GymEnvironment(args.env_name, log_path='./ppo_log_render', render=True, record=args.record)
    fps = env_render.env.metadata.get('video.frames_per_second', 30)
    logging.info(str(env))
    agent_name = env.external_agent_names[0]
    tf.reset_default_graph()
    ppo_model = models.create_agent_model(env,
                                          lr=args.learning_rate,
                                          h_size=args.hidden_units,
                                          epsilon=args.epsilon,
                                          beta=args.beta,
                                          max_step=args.max_steps,
                                          normalize=args.normalize_steps,
                                          num_layers=args.num_layers)
    is_continuous = env.agents[agent_name].action_space_type == 'continuous'
    use_observations = False
    use_states = True
    if not args.load_model:
        shutil.rmtree(args.summary_path, ignore_errors=True)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)
    tf.set_random_seed(np.random.randint(1024))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=args.keep_checkpoints)
    with tf.Session() as sess:
        # Instantiate model parameters
        if args.load_model:
            logging.info('model:loading')
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            if ckpt is None:
                logging.error('The model {0} could not be found. Make sure you specified the right --run-path'.format(args.model_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        steps, last_reward = sess.run([ppo_model.global_step, ppo_model.last_reward])
        summary_writer = tf.summary.FileWriter(args.summary_path)
        info = env.reset()[agent_name]
        trainer = tnr.Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states, args.train_model)
        trainer_monitor = tnr.Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states, False)
        render_started = False
        thread = None
        while steps <= args.max_steps or not args.train_model:
            if env.global_done:
                info = env.reset()[agent_name]
                trainer.reset_buffers(info, total=True)
            # Decide and take an action
            info = trainer.take_action(info, env, agent_name, steps, args.normalize_steps, stochastic=True)
            trainer.process_experiences(info, args.time_horizon, args.gamma, args.lambd)
            if len(trainer.training_buffer['actions']) > args.buffer_size and args.train_model:
                if args.render:
                    thread.pause()
                logging.info('model:optimizing')
                t = time.time()
                # perform gradient descent with experience buffer
                trainer.update_model(args.batch_size, args.num_epoch)
                logging.info('model:optimization completed:{:.2f} seconds'.format(float(time.time() - t)))
                if args.render:
                    thread.resume()
            if steps % args.summary_freq == 0 and steps != 0 and args.train_model:
                # write training statistics to tensorboard
                trainer.write_summary(summary_writer, steps)
            if steps % args.save_freq == 0 and steps != 0 and args.train_model:
                # save Tensorflow model
                models.save_model(sess=sess, model_path=args.model_path, steps=steps, saver=saver)
            if args.train_model:
                steps += 1
                sess.run(ppo_model.increment_step)
                if len(trainer.stats['cumulative_reward']) > 0:
                    mean_reward = np.mean(trainer.stats['cumulative_reward'])
                    sess.run(ppo_model.update_reward, feed_dict={ppo_model.new_reward: mean_reward})
                    last_reward = sess.run(ppo_model.last_reward)
            if not render_started and args.render:
                thread = renderthread.RenderThread(sess=sess,
                                                   trainer=trainer_monitor,
                                                   environment=env_render,
                                                   brain_name=agent_name,
                                                   normalize=args.normalize_steps,
                                                   fps=fps)
                thread.start()
                render_started = True
        # final Tensorflow model save
        if steps != 0 and args.train_model:
            models.save_model(sess=sess, model_path=args.model_path, steps=steps, saver=saver)
    env.close()
    models.export_graph(args.model_path, args.env_name)
    os.system('shutdown')
