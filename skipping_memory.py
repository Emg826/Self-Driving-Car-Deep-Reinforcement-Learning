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

  def __init__(self, limit, num_states_in_window, skip_factor):
    """
    :param limit: max number of states to store in memory
    :param num_states_in_window: number of states to stack
    :param skip_factor: only stack every skip_factor-th state; skip_factor=1 is
    the same as SequentialMemory, i.e., no states are skipped and
    num_states_in_window is equivalent to window_length
    """
    self.num_states_in_window = num_states_in_window
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
    Get a random batch of size batch_size from "memory"

    :param batch_size: size of mini batch to sample
    :param batch_idxs: ignored; is here only because keras-rl requires this
    param; this func generates its own indices

    :returns: list of transitions of the form
    """
    num_entries = self.nb_entries
    assert num_entries >= (self.num_states_in_window * self.skip_factor)

    # note: note sampling samples consecutively, literally just
    # drawing batch_size number of indicies w/out replacement (as per keras-rl
    # implementation)
    sample_indices = sample(range(self.num_states_in_window*self.skip_factor-1,
                                  num_entries-1),
                            size=batch_size)

    experiences = []
    # note sample_idx=t, so state_t+1 is located @ (sample_idx+1 % num_entries)
    # question: do i also apply skips to the statet+1 idx? i'd say no.
    for sample_idx in sample_indices:
      # 1. see if terminal idx in any of previous indices (need to check in
      # between skips just in case episode on a transition that is not stacked)
      attempts_at_new_valid_idx = 0
      max_num_retries_at_valid_idx = 10
      if episode terminates between last stack state and state_t+1
      # note: 2 cases, statet+1 is new episode, and somewhere b4 state_t but
      # after last-most stacked state is new episode
      while attempts_at_new_valid_idx < max_num_retries_at_valid_idx:
      # 2.



    end_idx_of_batch = randint(0+,
                               num_entries-1)
    sample_count = 0
    experiences = []

    while sample_count < batch_size:


    # note: this way, end idx can be > self.limit; handle this in list
    # comprehension by % num_entries

    sample_end_idx = sample_start_idx + self.num_states_in_window * self.skip_factor

    batch_idxs = [idx % num_entries for idx in range(sample_start_idx,
                                                     sample_end_idx,
                                                     self.skip_factor)]

    experiences = []  # transitions, list of 'Experience' tuples

    # time step t is terminal, then for state_t+1, just use state_t for simplicity
    # note: i don't think this actually matters for deep Q learning since
    # phi_t+1/state_t+1 would only be used in calculating reward and even then,
    # only when not in terminal state (if terminal, then target is r_t)
      # note, what i did is not good enough


    # 2 edge cases: wrap around at last insert (since last insert and then idx
    # right after that are 2 different episdoes) and wrap around at end of list

    # problem: since skipping indices, what if stack state from prev episode?







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
    return {'window_length': self.num_states_in_window,
            'ignore_episode_boundaries': False,
            'limit': self.limit}
