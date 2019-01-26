"""
DQNAgent, but the q-values are printed each time an input is
fed forward through the DQN
"""

from rl.agents.dqn import DQNAgent


class TransparentDQNAgent(DQNAgent):
  def __init__(self, print_frequency, *args, **kwargs):
    # print the Q-values every print_frequency steps (>= 1, int)
    super(TransparentDQNAgent, self).__init__(*args, **kwargs)

    self.print_frequency = print_frequency
    self.forward_passes_since_last_print = 0 

  def forward(self, observation):
    # Select an action.

    state = self.memory.get_recent_state(observation)

    q_values = self.compute_q_values(state)

    if self.training:
      action = self.policy.select_action(q_values=q_values)
    else:
      action = self.test_policy.select_action(q_values=q_values)

    # Kowalski! Status report.
    self.forward_passes_since_last_print += 1
    if self.print_frequency == self.forward_passes_since_last_print:
      # put max value in []
      printable_q_values = list(q_values.copy())
      
      max_q_value = -1000000
      max_q_value_idx = -1

      for idx, q_value in enumerate(printable_q_values):
        if max_q_value < q_value:
          max_q_value = q_value
          max_q_value_idx = idx

      printable_q_values[max_q_value_idx] = [max_q_value]
    
      print(printable_q_values)
      self.forward_passes_since_last_print = 0

    # Book-keeping.
    self.recent_observation = observation

    self.recent_action = action

    return action
