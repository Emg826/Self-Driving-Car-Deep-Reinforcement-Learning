"""
Helpful in creating this class:
https://github.com/keras-rl/keras-rl/blob/ac1439e7029b7d514aca53d3eecec5f64193d949/rl/agents/dqn.py#L33
https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py
https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
https://docs.python.org/2/library/collections.html#collections.deque
""""

from rl.memory import Memory, sample_batch_indexes
from collections import namedtuple, deque # pronounced "deck"
from random import sample

# as per keras-rl/rl/memory.py
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class SkippingMemory():
  """
  For when WINDOW_LENGTH > 1 but want every nth state from memory. Note, n=1
  is just SequentialMemory. "Skips" over all but every nth state until
  num_states_to_stack number of states have been retrieved.

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
    self.skip_factor = skip_factor  # only stack every skip_factorth frame
    self.window_length = (self.num_states_to_stack * self.skip_factor) - 1

    self.recent_states = deque(maxlen=self.window_length+self.skip_factor) # data struct keras-rl uses
    self.recent_terminals = deque(maxlen=self.window_length+self.skip_factor) # right-end is most recent

    # note: stores transitions as in sequential memory, i.e., consecutively
    self.actions = [None] * self.limit
    self.rewards = [None] * self.limit
    self.terminals = [None] * self.limit
    self.states = [None] * self.limit

  def get_recent_state(self, current_observation):
    """
    Return a list of the last state (state will be stacked as you would expect).
    Called in forward() method of DQN, i.e., when want the DQN to output
    Q values in a given state

    :param current_observation: the most recent observation
    """

    # stack states as expected except when terminal within window (new episode
    # between one stacked image and the next). then, just append the last valid
    # state the rest of the times.
    recent_state = [current_observation]  # list is ok since expect channel_first param in keras neural network

    # if there aren't enough states to stack, then just stack the current one.
    # lazy since could could have <  and be able to stack, but not big deal enough to worry about that
    if self.num_observations <= self.window_length:
      return [current_observation] * self.num_states_to_stack

    next_recent_state_idx_to_stack = len(self.recent_states) - self.skip_factor
    last_valid_state = current_observation
    list_of_recent_terminals = list(self.recent_terminals)
    while len(recent_state) < self.num_states_to_stack:

      # check if state @ next_recent_state_idx is the state_t+1 in the last transition
      # of the prior episode
      next_recent_state_is_from_previous_episode = False
      some_state_after_next_recent_state_is_from_next_episode = False
      if next_recent_state_idx_to_stack - 1 >= 0:
        next_recent_state_is_from_previous_episode = self.recent_terminals[next_recent_state_idx_to_stack - 1]
        some_state_after_next_recent_state_is_from_next_episode = True in list_of_recent_terminals[next_recent_state_idx_to_stack:]

      # if not enough states in current episode to stack or if ran out of states to stack (for some reason?)
      if next_recent_state_is_from_previous_episode is True or \
         next_recent_state_idx_to_stack < 0 or \
         some_state_after_next_recent_state_is_from_next_episode is True:
        # then just make rest of stacked states the last valid one
        while len(recent_state) < self.num_states_to_stack:
          recent_state.insert(0, last_valid_state)
        return recent_state
      # else, the next state to stack is in same epsiode as current_observation; OK to stack
      else:
        # idk why stack past--to-->current, but keras-rl does this
        recent_state.insert(0, self.recent_states[next_recent_state_idx_to_stack])
        
        last_valid_state = self.recent_states[next_recent_state_idx_to_stack]
        next_recent_state_idx_to_stack -= self.skip_factor

    return recent_state 

  def sample(self, batch_size, batch_idxs=None):
    """
    Get a random batch of size batch_size from "memory". Used during training.

    :param batch_size: size of mini batch to sample
    :param batch_idxs: ignored; is here only because keras-rl requires this
    param; this func generates its own indices

    :returns: list of Experience tuples
    """
    num_entries = self.nb_entries
    assert num_entries-1 >= (self.window_length)

    experiences = []
    while len(experiences) < batch_size:
      # 1. Get a random index -- sample idx_t+1 because SequentialMemory does it
      idx_t_plus_1 = sample_batch_indexes(self.window_length+1, num_entries, size=1)[0]

      # 2. Check that idx_t is not final state in an episode, e.g., it's not
      # state_t+1 in the terminal transition: (state_t, action_t, reward_t, state_t+1)
      terminal_t_minus_1 = self.terminals[idx_t_plus_1 - 2]  # boolean

      # 3. if t-1 is terminal, then that means state_t is the 'next state' in the
      # terminal transition tuple ==> state_t is final state of an epsiode, not
      # the first state of a new episode
      while terminal_t_minus_1:
        idx_t_plus_1 = sample_batch_indexes(self.window_length+1, num_entries, size=1)[0]
        terminal_t_minus_1 = self.terminals[idx_t_plus_1 - 2]

      # at this point, know for sure that state_t is not the end of an episode.
      # so, state_t+1 is in the same episode as state_t

      # 4. now need to check and see if any of the states in the window are
      # terminal since this would imply that one stacked frame is in a different
      # episode than another stacked frame
      idx_t = idx_t_plus_1 - 1
      if self.terminal_within_window(idx_t):
        continue  # don't bother if have a terminal in window; just start over

      # 5. stack the states
      state_t = [] # stacking states, so need a list to store multiple states
      state_t_plus_1 = [] # also have to stack state_t+1 use stacked inputs to compute targets
      idx = idx_t
      while len(state_t) < self.num_states_to_stack:
        state_t.append(self.states[idx])
        state_t_plus_1.append(self.states[(idx+1)%num_entries])
        idx = (idx - self.skip_factor) % num_entries

      # 6. Build Experience tuple
      experiences.append(Experience(state0=state_t,
                                    action=self.actions[idx_t],
                                    reward=self.rewards[idx_t],
                                    state1=state_t_plus_1,
                                    terminal1=self.terminals[idx_t]))
      
    assert len(experiences) == batch_size
    
    return experiences

  def append(self, observation, action, reward, terminal, training=True):
    """
    Append state, action, reward, terminal to "memory" lists

    :param state: state for time step t
    :param action: action taken in time step t
    :param reward: reward for taking action in time step t
    :param terminal: True if time step t was last time step in episode; else False
    """
    self.recent_states.append(observation)
    self.recent_terminals.append(terminal)
    self.num_observations += 1

    # check if training  because if testing, then don't need replay memory
    if training is True:
      # do % for when num_observations > limit ... num_observations is
      # cumulative/total num in whole training session
      idx_to_insert_at = self.num_observations % self.limit
      self.states[idx_to_insert_at] = observation
      self.actions[idx_to_insert_at] = action
      self.rewards[idx_to_insert_at] = reward
      self.terminals[idx_to_insert_at] = terminal

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

  def terminal_within_window(self, idx_t):
    """
    Check if there are any terminal transitions in window of sample_idx

    :param idx_t: idx of state_t you are sampling
    :returns: False if termination in window - else True
    """
    num_entries = self.nb_entries

    # could be negative; handle w/ % in []
    least_idx = idx_t - self.window_length

    # note: to check if state_i is a terminal state, need to check if idx_i-1
    # has terminal, which s
    indices_in_window = [idx % num_entries for idx in range(least_idx-1, idx_t-1)]

    # check if any terminals in window (including @ states not stacked)
    for idx in indices_in_window:
      if self.terminals[idx] is True:
        return True

    return False
