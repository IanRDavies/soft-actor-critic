import numpy as np


class ReplayBuffer(object):
    '''
    Standard FIFO Replay Buffer implementation using NumPy

    Arguments
    obs_dim - int
        The dimensionality of observations
    action_dim - int
        The dimension of action vectors
    max_size - int
        The maximum number of time steps of data the replay
        buffer can hold. Stale data is overwritten
    '''

    def __init__(self, obs_dim, action_dim, max_size):
        '''Initialisation of replay buffer'''
        # Separate storage for each variable to track.
        # Stored in private attributes
        self._obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self._next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._rewards = np.zeros((max_size,), dtype=np.float32)
        self._terminals = np.zeros((max_size,), dtype=np.float32)
        # Track the amount of data stored in the replay buffer
        self._size = 0
        # Track where to insert each new item
        self._insert_index = 0
        # Keep the capacity as an attribute
        self.max_size = max_size

    def __len__(self):
        '''Compatibility with the built-in len function'''
        return self._size

    def add(self, obs, act, rew, next_obs, terminal):
        '''
        Add a new time step of experience to the buffer.

        Arguments
        obs - np.array
            The agent's observation
        act - np.array
            The action the agent chose based on the current observation
        rew - float
            The reward received from the environment after action executed
        next_obs - np.array
            The subsequent observation
        terminal - float or bool
            The end of episode flag
        '''
        # Store each piece of experience provided in the relevant array
        self._obs[self._insert_index] = obs
        self._actions[self._insert_index] = act
        self._rewards[self._insert_index] = rew
        self._next_obs[self._insert_index] = next_obs
        self._terminals[self._insert_index] = terminal
        # Update the size of the replay buffer
        self._size = min(self._size + 1, self.max_size)
        # Increment the insertion index modulo buffer capacity
        self._insert_index = (self._insert_index + 1) % self.max_size

    def sample(self, batch_size):
        '''
        Sample a minibatch of experience from the replay buffer.

        Arguments
        batch_size - int
            The number of time steps of experience to sample

        Returns
        obs - np.array
            Array of size (batch_size, obs_dim) of observations
        actions - np.array
            Array of size (batch_size, action_dim) of actions
        rewards - np.array
            Array of size (batch_size, ) of reward scalars
        next_obs - np.array
            Array of size (batch_size, obs_dim) of subsequent
            observations
        terminals - np.array
            Array of size (batch_size, ) of termination flags
        '''
        # First sample the indices of the experience to return
        indices = np.random.randint(0, self._size, size=batch_size)
        # Use the sampled indices to get the relevant data and return it
        return (self._obs[indices], self._actions[indices], self._rewards[indices],
                self._next_obs[indices], self._terminals[indices])
