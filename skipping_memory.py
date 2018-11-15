from rl.memory import Memory
from collections import namedtuple
from random import sample

# as per keras-rl/rl/memory.py
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class SkippingMemory():
  """
  For when WINDOW_LENGTH > 1 but want every nth state from memory. Note, n=1
  is just SequentialMemory. "Skips" over all but every nth state until
  window length number of states have been retrieved.

  Note: this class is meant to fit into the keras-rl module, so it is modeled
  after the SequentialMemory class from v0.4.2
  """

  def __init__(self, limit, num_states_to_stack, skip_factor):
    """
    :param limit: max number of states to store in memory
    :param num_states_to_stack: number of states to stack
    :param skip_factor: only stack every skip_factor-th state; skip_factor=1 is
    the same as SequentialMemory, i.e., no states are skipped and
    num_states_to_stack is equivalent to window_length
    """
    self.num_states_to_stack = num_states_to_stack
    self.limit = limit
    self.num_observations = 0

    assert skip_factor > 0, "Need skip factor > 0. Note: skip_factor=1 is same as SequentialMemory"
    self.skip_factor = skip_factor

    # note: stores transitions as in sequential memory, i.e., consecutively
    self.actions = [None] * self.limit
    self.rewards = [None] * self.limit
    self.terminals = [None] * self.limit
    self.states =  = [None] * self.limit

  def sample(self, batch_size, batch_idxs=None):
    """
    Get a random batch of size batch_size from "memory". Note, this is somewhat
    inefficient since could note where episode split is

    :param batch_size: size of mini batch to sample
    :param batch_idxs: ignored; is here only because keras-rl requires this
    param; this func generates its own indices

    :returns: list of transitions of the form
    """
    num_entries = self.nb_entries
    assert num_entries-1 >= (self.num_states_to_stack * self.skip_factor)

    # note: note sampling samples consecutively, literally just
    # drawing batch_size number of indicies w/out replacement (as per keras-rl
    # implementation)
    sample_idx_range == range(0, num_entries-1)

    experiences = []
    sample_idx = None
    while len(experiences) < batch_size:
      # 1.  Get a random index
      if sample_idx is None:
        sample_idx = sample(sample_idx_range, size=1)
      # else: already have sample_idx from previous iteration

      # 2. if sample is terminal, i.e., next state is in diff epsidoe
      if self.terminals[sample_idx]:
        sample_idx = (sample_idx - 1) % num_entries
        continue

      # 3. Check if this is a terminating transition or if there is a
      # terminating transition anywhere in this window's past
      termination_within_window, idx_of_terminal = self._termination_within_window(sample_idx)
      if termination_within_window:
        sample_idx = None  # not going to sample_idx - idx_of_terminal in
        continue
        # case could get stuck in loop of resetting to same sample indicies

      # 4. Build Experience tuple
      t_idx = sample_idx
      stack_t_indices = 0
      stack_t_indices = 1
      experiences.append(Experience(state0=self.states[stack_t_indices],
                                    action=self.actions[t_idx],
                                    reward=self.rewards[t_idx],
                                    state1=self.states[stack_t_indices],
                                    terminal1=self.terminals[t_idx]))
      sample_idx = None  # signals successful sampling, so need new rand idx






  def append(self, state, action, reward, terminal, training=True):
    """
    Append state, action, reward, terminal to "memory" lists

    :param state: state for time step t
    :param action: action taken in time step t
    :param reward: reward for taking action in time step t
    :param terminal: True if time step t was last time step in episode; else False
    """
    # not sure why SequentialMemory has this check, but it does, so I'm doing it too
    if training is True:
      # do % for when num_observations > limit ... num_observations is
      # cumulative/total num in whole training session
      idx_to_insert_at = self.num_observations % self.limit
      self.states[idx_to_insert_at] = state
      self.actions[idx_to_insert_at] = action
      self.rewards[idx_to_insert_at] = reward
      self.terminal[idx_to_insert_at] = terminal

      self.num_observations += 1

  @property
  def nb_entries(self):
    """
    Return number of states in self.states.
    """
    return min(self.num_observations, self.limit)

  def get_config(self):
    """
    Return the configurations of SkippingMemory, as per this function in
    SequentialMemory class.
    """
    return {'window_length': self.num_states_to_stack,
            'ignore_episode_boundaries': False,
            'limit': self.limit}

  def _termination_within_window(self, sample_idx):
    """
    Check if there are any terminal transitions in window of sample_idx

    :param sample_idx: idx of state_t you are sampling
    :returns: tuple: False if termination in window, else True;
                     idx of episode termination
    """
    num_entries = self.nb_entries

    # can be negative; handle w/ % in
    lower_idx = sample_idx - self.num_states_to_stack * self.skip_factor

    indices_in_window = [idx % num_entries for idx in range(lower_idx, sample_idx)]

    # for each idx in window (including indices not going to be stacked),
    # check if terminal
    for idx in indices_in_window:
      if self.terminals[idx] is True:
        return True, idx

    return False, None
