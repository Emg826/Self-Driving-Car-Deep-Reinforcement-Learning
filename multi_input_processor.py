import numpy as np

from rl.core import Processor


class MultiInputProcessor(Processor):
  """
  [ [input1_t1, input2_t1, input3_t1], [input1_t2, input2_t2, input3_t2], ...] ]

  to

  [ [input1_t1, input1_t2, ...], [input2_t1, input2_t2, ...], [input3_t1, input3_t2], ... ],
  which is what a multi-input neural network expects (at least, keras' MI neural networks)
  """
  def __init__(self, num_inputs):
    self.num_inputs = num_inputs

  def process_state_batch(self, state_batch):
    """
    Assumes that each of these inputs is already a numpy array, i.e.,
    it could be fed into a neural network on its own.

    """
    # handle the feed fwd case where, for whatever reason, keras-rl's
    # dqn compute_batch function, when it calls process_batch, wraps the batch
    # in a numpy array
    if len(state_batch.shape) == 3 and state_batch.shape[0] == 1:
      state_batch = state_batch[0]
    
    state_batch_separated_by_input = [[] for x in range(self.num_inputs)]

    # for 1 of n samples in minibatch (which'll be a list)
    for state in state_batch:
        for input_idx, input_data in enumerate(state):     
          state_batch_separated_by_input[input_idx].append(input_data)

    return state_batch_separated_by_input
