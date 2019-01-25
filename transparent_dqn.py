"""
DQNAgent, but the q-values are printed each time an input is
fed forward through the DQN
"""

from rl.agents.dqn import DQNAgent


class TransparentDQNAgent(DQNAgent):
  def __init__(self, *args, **kwargs):
    super(TransparentDQNAgent, self).__init__(*args, **kwargs)

  def forward(self, observation):
    # Select an action.

    state = self.memory.get_recent_state(observation)

    q_values = self.compute_q_values(state)
    print(q_values)

    if self.training:
      action = self.policy.select_action(q_values=q_values)

    else:
      action = self.test_policy.select_action(q_values=q_values)

    # Book-keeping.
    self.recent_observation = observation

    self.recent_action = action

    return action
