"""

Contains an implementation of PPO as described here: https://arxiv.org/abs/1707.06347

"""
import environment as environ
import logging
import model_utils
import numpy as np
import os
import renderthread as ppo_thread
import tensorflow as tf
import time
import trainer as ppo_tnr

def ppo(args):
    env = environ.GymEnvironment(args.dir_path,
                                 args.env_name,
                                 'ppo_log',
                                 args.model_name,
                                 skip_frames=6)
    env_render = environ.GymEnvironment(args.dir_path,
                                        args.env_name,
                                        'ppo_log_render',
                                        args.model_name,
                                        record=True,
                                        render=args.render)
    fps = env_render.env.metadata.get('video.frames_per_second', 30)
    logging.info(str(env))
    brain_name = env.external_brain_names[0]
    tf.reset_default_graph()
    ppo_model = model_utils.create_agent_model(env,
                                               args.beta,
                                               args.epsilon,
                                               args.hidden_units,
                                               args.learning_rate,
                                               args.max_steps,
                                               args.normalize_steps,
                                               args.num_layers)
    is_continuous = env.brains[brain_name].action_space_type == 'continuous'
    use_observations = False
    use_states = True
    model_path = os.path.join(args.dir_path, 'models', args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    tf.set_random_seed(np.random.randint(1024))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=args.keep_checkpoints)
    with tf.Session() as sess:
        # Instantiate model parameters
        if args.load_model:
            logging.info('model:loading')
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                logging.error('model could not be found:{}'.format(model_path))
            saver.restore(sess, ckpt.model_checkpoint_path)    # pylint:disable=E1101
        else:
            sess.run(init)
        steps, _ = sess.run([ppo_model.global_step, ppo_model.last_reward])
        summary_writer = tf.summary.FileWriter(model_path)
        info = env.reset()[brain_name]
        trainer = ppo_tnr.Trainer(ppo_model,
                                  sess,
                                  info,
                                  is_continuous,
                                  use_observations,
                                  use_states,
                                  args.train_model)
        trainer_monitor = ppo_tnr.Trainer(ppo_model,
                                          sess,
                                          info,
                                          is_continuous,
                                          use_observations,
                                          use_states,
                                          training=False)
        render_started = False
        while steps <= args.max_steps or not args.train_model:
            if env.global_done:
                info = env.reset()[brain_name]
                trainer.reset_buffers(info, total=True)
            if not render_started and args.render:
                renderthread = ppo_thread.RenderThread(sess,
                                                       trainer_monitor,
                                                       env_render,
                                                       brain_name,
                                                       args.normalize_steps,
                                                       fps)
                renderthread.start()
                render_started = True
            info = trainer.take_action(info, env, brain_name, steps, args.normalize_steps)
            trainer.process_experiences(info, args.time_horizon, args.gamma, args.lambd)
            if len(trainer.training_buffer['actions']) > args.buffer_size and args.train_model:
                if args.render:
                    renderthread.pause()
                logging.info('optimization:started')
                t = time.time()
                # perform gradient descent with experience buffer
                trainer.update_model(args.batch_size, args.num_epoch)
                logging.info('optimization:finished:duration:{:.1f} seconds'.format(float(time.time() - t)))
                if args.render:
                    renderthread.resume()
            if steps % args.summary_freq == 0 and steps != 0 and args.train_model:
                trainer.write_summary(summary_writer, steps)
            if steps % args.save_freq == 0 and steps != 0 and args.train_model:
                model_utils.save_model(sess, saver, model_path, steps)
            if args.train_model:
                steps += 1
                sess.run(ppo_model.increment_step)
                if len(trainer.stats['cumulative_reward']) > 0:
                    mean_reward = np.mean(trainer.stats['cumulative_reward'])
                    sess.run(ppo_model.update_reward, {ppo_model.new_reward: mean_reward})
                    _ = sess.run(ppo_model.last_reward)
        # final save Tensorflow model
        if steps != 0 and args.train_model:
            model_utils.save_model(sess, saver, model_path, steps)
    env.close()
    model_utils.export_graph(model_path, args.env_name)
    os.system('shutdown')
