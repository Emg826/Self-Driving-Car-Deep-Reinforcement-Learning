"""https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py"""
"""https://github.com/Kjell-K/AirGym/blob/master/DQN-Train.py"""
"""https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras"""

import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint  # https://github.com/keras-rl/keras-rl/blob/171667dce2a39993705b12fdf0b3cc3bb7bf26d2/rl/callbacks.py

from airsim_env import AirSimEnv
from skipping_memory import SkippingMemory


env = AirSimEnv(num_steering_angles=5,
                       max_num_steps_in_episode=10**4,
                       time_steps_between_dist_calc=40)
num_steering_angles = env.action_space.n

random.seed()
np.random.seed()

INPUT_SHAPE = (260-int(3*260/7), 770) # H x W (no channels because assume DepthPlanner)
NUM_FRAMES_TO_STACK = 7  # reward_delay * this = prev sec as input
STACK_EVERY_N_FRAMES = 3


input_shape = (NUM_FRAMES_TO_STACK,) + INPUT_SHAPE

model = Sequential()
model.add(Conv2D(32, kernel_size=5, strides=4,
                 input_shape=input_shape, data_format = 'channels_first'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.add(Conv2D(64, kernel_size=4, strides=3))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.add(Flatten())
model.add(Dense(64, activity_regularizer=l2(0.03)))
model.add(Activation('elu'))

model.add(Dense(96, activity_regularizer=l2(0.03)))
model.add(Activation('elu'))

model.add(Dense(num_steering_angles))
print(model.summary())


#replay_memory = SequentialMemory(limit=10**4, NUM_FRAMES_TO_STACK=NUM_FRAMES_TO_STACK)
replay_memory = SkippingMemory(limit=10**4,
                                              num_states_to_stack=NUM_FRAMES_TO_STACK,
                                              skip_factor=STACK_EVERY_N_FRAMES)

# something like: w/ probability epsilon (which decays through training),
# select a random action; otherwise, consult the agent
# epsilon = f(x) = ((self.value_max - self.value_min) / self.nb_steps)*x + self.value_max
num_total_training_steps = 10**5

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0, # start off 100% random
                              value_min=0.05,  # get to random action x% of time
                              value_test=0.0,  # when testing, take rand action this val *100 % of time
                              nb_steps=num_total_training_steps) # of time steps to go from epsilon=value_max to =value_min


ddqn_agent = DQNAgent(model=model, nb_actions=num_steering_angles,
                     memory=replay_memory, enable_double_dqn=True,
                     enable_dueling_network=False, target_model_update=1e-1, # soft update parameter?
                     policy=policy, gamma=0.99, train_interval=4,
                     nb_steps_warmup=10**3)

ddqn_agent.compile(Adam(lr=1e-4), metrics=['mae']) # not use mse since |reward| <= 1.0

weights_filename = 'ddqn_collision_avoidance_1115.h5'
want_to_train = True
train_from_weights_in_weights_filename = True

if want_to_train is True:

  # note: interval's units are episode_steps
  callbacks_list = [ModelIntervalCheckpoint(filepath=weights_filename, verbose=5, interval=1000)]

  if train_from_weights_in_weights_filename:
    try:
      ddqn_agent.load_weights(weights_filename)
      print('Successfully loaded DDQN weights')
    except:
      print('Failed to load DDQN weights')

  ddqn_agent.fit(env, callbacks=callbacks_list, nb_steps=num_total_training_steps,
                      visualize=False, verbose=2)

  ddqn_agent.save_weights(weights_filename)
else: # else want to test
    ddqn_agent.load_weights(weights_filename)
    ddqn_agent.test(env, nb_episodes=12, visualize=True)
