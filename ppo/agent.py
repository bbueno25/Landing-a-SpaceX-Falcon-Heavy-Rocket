class AgentInfo:

    def __init__(self, state, memory=None, reward=None, agents=None, local_done=None, action=None):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.agents = agents
        self.local_done = local_done
        self.memories = memory
        self.previous_actions = action
        self.rewards = reward
        self.states = state

class AgentParameters:

    def __init__(self, name, params):
        """
        Contains all brain-specific parameters.

        :param name: Name of brain.
        :param params: Dictionary of brain parameters.
        """
        self.name = name
        self.action_space_size = params['actionSize']
        self.action_space_type = params['actionSpaceType']
        self.number_observations = 0
        self.state_space_size = params['stateSize']
        self.state_space_type = params['stateSpaceType']

    def __str__(self):
        params = {'unity name': self.name,
                  'state space type': self.state_space_type,
                  'state space size (per agent)': str(self.state_space_size),
                  'action space type': self.action_space_type,
                  'action space size (per agent)': str(self.action_space_size)}
        return params
