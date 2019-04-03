import numpy as np

from rl.core import Processor
import time


class MultiInputProcessor(Processor):
  """
  [ [input1_t1, input2_t1, input3_t1], [input1_t2, input2_t2, input3_t2], ...] ]

  to

  [ [input1_t1, input1_t2, ...], [input2_t1, input2_t2, ...], [input3_t1, input3_t2], ... ],
  which is what a multi-input neural network expects (at least, keras' MI neural networks)
  """
  def __init__(self, num_inputs, num_inputs_stacked):
    self.num_inputs = num_inputs
    self.num_inputs_stacked = num_inputs_stacked

  def process_state_batch(self, state_batch):
    """
    Assumes that each of these inputs is already a numpy array, i.e.,
    it could be fed into a neural network on its own.
    """
    # handle the feed fwd case where, for whatever reason, keras-rl's
    # dqn compute_batch function, when it calls process_batch, wraps the batch
    # in a numpy array
    #print(state_batch.shape)  # debug

    state_batch_separated_by_input = [[] for x in range(self.num_inputs)]
    if self.num_inputs_stacked == 1:

      if len(state_batch.shape) == 3 and state_batch.shape[0] == 1:
        state_batch = state_batch[0]
    

      # for 1 of n samples in minibatch (which'll be a list)
      for state in state_batch:
        for input_idx, input_data in enumerate(state):     
          state_batch_separated_by_input[input_idx].append(input_data)
    else:
      for state_in_batch in state_batch:
        # first num_inputs_stacked states in state_batch are actually
        # 1 state; skipping memory just never concatenated them into 1 list
        for start_of_this_input_stack_idx in range(0, len(state_in_batch), self.num_inputs_stacked):
          input_stack = state_in_batch[start_of_this_input_stack_idx:(start_of_this_input_stack_idx + self.num_inputs_stacked)]
          # input_stack: (self.num_inputs_stacked, num_inputs kinds of inputs)
          # as far as what we want to append to state_batch_separated_by_input,
          # we want to make num_inputs number of appends. instead of appending
          # a single image or a 1D array of sensor data to each of the num_inputs
          # lists in state_batch_sep, we want to append a list num_inputs_stacked long
          
          #print(input_stack[0].shape, input_stack[1].shape)  # debug

          # for each kind of the num_inputs (3) inputs -- a col idx
          for idx_of_current_input_type in range(0, self.num_inputs):
            stack_of_current_type_of_input = []

            # for each of the num_inputs_stacked (stack size) arrays of this kind in all of input_stack
            # a row idx
            for list_with_num_input_many_things in input_stack:
              stack_of_current_type_of_input.append(list_with_num_input_many_things[idx_of_current_input_type])

            # now for the current input type, it is all stacked accordingly, so append
            # to state_batch_sep
            state_batch_separated_by_input[idx_of_current_input_type].append(stack_of_current_type_of_input)

    #print(len(state_batch_separated_by_input),len(state_batch_separated_by_input[0]))  # debug

      
    return state_batch_separated_by_input
