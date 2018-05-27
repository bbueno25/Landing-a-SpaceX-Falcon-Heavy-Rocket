import atexit
import brain
import gym
import gym_vertical_landing
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

class GymEnvironment:

    def __init__(self,
                 dir_path,
                 env_name,
                 log_path,
                 model_name,
                 record=False,
                 render=False,
                 skip_frames=1):
        atexit.register(self.close)
        self._academy_name = 'VerticalLandingAcademy'
        self._current_returns = {}
        self._last_action = []
        self.render = render
        self.env = gym.make(env_name)
        if skip_frames < 0 or not isinstance(skip_frames, int):
            logger.error('Invalid frame skip value. Frame skip deactivated.')
        elif skip_frames > 1:
            # frameskip_wrapper = gym.wrappers.(skip_frames)
            # self.env = frameskip_wrapper(self.env)
            self.env = self.env
        if record:
            self.env = gym.wrappers.Monitor(self.env,
                                            directory=os.path.join(dir_path, 'video', model_name),
                                            force=True,
                                            video_callable=lambda x: x % 1 == 0)
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        if isinstance(ac_space, gym.spaces.Box):
            assert len(ac_space.shape) == 1
            self.ac_space_type = 'continuous'
            self.ac_space_size = ac_space.shape[0]
        elif isinstance(ac_space, gym.spaces.Discrete):
            self.ac_space_type = 'discrete'
            self.ac_space_size = ac_space.n
        else:
            raise NotImplementedError
        if isinstance(ob_space, gym.spaces.Box):
            assert len(ob_space.shape) == 1
            self.ob_space_type = 'continuous'
            self.ob_space_size = ob_space.shape[0]
        elif isinstance(ob_space, gym.spaces.Discrete):
            self.ob_space_type = 'discrete'
            self.ob_space_size = ob_space.n
        else:
            raise NotImplementedError
        self._data = {}
        self._log_path = log_path
        self._global_done = False
        self._brains = {}
        self._brain_names = ['FirstBrain']
        self._external_brain_names = self._brain_names
        self._parameters = {'actionSize': self.ac_space_size,
                            'actionSpaceType': self.ac_space_type,
                            'stateSize': self.ob_space_size,
                            'stateSpaceType': self.ob_space_type}
        self._brains[self._brain_names[0]] = brain.BrainParameters(self._brain_names[0], self._parameters)
        self._loaded = True
        logger.info('environment started successfully.')

    def __str__(self):
        return str({'academy_name': self._academy_name,
                    'action_space_size': self.ac_space_size,
                    'action_space_type': self.ac_space_type,
                    'obs_space_size': self.ob_space_size,
                    'obs_space_type': self.ob_space_type})

    def _state_to_info(self):
        state = np.array(self._current_returns[self._brain_names[0]][0])
        if self.ob_space_type == 'continuous':
            states = state.reshape((1, self.ob_space_size))
        else:
            states = state.reshape((1, 1))
        memories = []
        rewards = [self._current_returns[self._brain_names[0]][1]]
        agents = [self._brain_names[0]]
        dones = [self._current_returns[self._brain_names[0]][2]]
        actions = self._last_action
        self._data[self._brain_names[0]] = brain.BrainInfo(states, actions, agents, dones, memories, rewards)
        return self._data

    @property
    def academy_name(self):
        return self._academy_name

    @property
    def brain_names(self):
        return self._brain_names

    @property
    def brains(self):
        return self._brains

    def close(self):
        """
        Sends a shutdown signal to the gym environment.
        """
        self.env.close()

    @property
    def external_brain_names(self):
        return self._external_brain_names

    @property
    def global_done(self):
        return self._global_done

    @property
    def logfile_path(self):
        return self._log_path

    def reset(self):
        """
        Sends a signal to reset the environment.

        return:
            Data structure corresponding to the
            initial reset state of the environment.
        """
        obs = self.env.reset()
        self._current_returns = {self._brain_names[0]: [obs, 0, False]}
        self._last_action = [0] * self.ac_space_size if self.ac_space_type == 'continuous' else 0
        self._global_done = False
        return self._state_to_info()

    def step(self, action=None):
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly,
        and returns observation, state, and reward information to the agent.

        action:
            Agent's action to send to environment.
            Can be a scalar or vector of int/floats.
        return:
            A Data structure corresponding to the new state of the
            environment.
        """
        action = {} if action is None else action
        if self._loaded and not self._global_done and self._global_done is not None:
            obs, rew, done, _ = self.env.step(action[0])
            if done:
                self._global_done = True
            self._current_returns[self._brain_names[0]] = [obs, rew, done]
            self._last_action = action
            if self.render:
                self.env.render()
            return self._state_to_info()
        elif not self._loaded:
            logging.error('No Gym environment is loaded.')
        elif self._global_done:
            logging.info('The episode is completed. Reset the environment with reset function.')
        elif self.global_done is None:
            logging.error('You cannot conduct step without first calling reset. Reset the environment with reset function.')
