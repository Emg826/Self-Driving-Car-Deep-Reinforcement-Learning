"""https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py"""
"""https://github.com/Kjell-K/AirGym/blob/master/DQN-Train.py"""
"""https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras"""

""" Copy and paste this into your settings.json (which hould be in your Documents folder)

{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "RpcEnabled": true,
  "ViewMode": "SpringArmChase",
  "EngineSound": false,
  "ClockSpeed": 1.0,
  "CameraDefaults": {
    "CaptureSettings": [{
      "ImageType": 0,
      "Width": 640,
      "Height": 384,
      "FOV_Degrees": 120
    },
    {
      "ImageType": 1,
      "Width": 640,
      "Height": 384,
      "FOV_Degrees": 120
    }]
  }           
}

"""
# 5.0 steps per IRL second in NoDisplay and 3.77 in SpringArmChase, trying to inrc clockspeed to 1.25 from 1.0
# the idea is that @ 1.0 speed and SpringArmChase, want same number of steps per IRL second as \
# > 1.0 speed?
# @ 1.5 cspeed, get about 5.38 steps per IRL second; need >= 5.6
# 1.4 * 3.77 needed 5.3 and is getting ~ that, so 1.4 works , i.e., same ratio of in game steps to in game
# time?    a proportion   3.77/ 1.0   = x / 1.4

import numpy as np
import random
import math

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


# This block solved the "CUBLAS_STATUS_ALLOC_FAILED" CUDA issue (https://stackoverflow.com/a/52762075)
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))

# have to be careful not to make PHI too complex, else decr num steps per IRL second
PHI = lambda pixel: min(255.0, 2048 / pixel)


env = AirSimEnv(num_steering_angles=3,
                      max_num_steps_in_episode=10**4,
                      settings_json_image_w=640,  # from settings.json
                      settings_json_image_h=384,
                      fraction_of_top_of_img_to_cutoff=0.37,
                      fraction_of_bottom_of_img_to_cutoff=0.47,
                      seconds_pause_between_steps=0.00,  # so as to prevent extreme case of 1000 steps per second (if that was possible)
                      seconds_between_collision_in_sim_and_register=2.5,  # note avg 4.12 steps per IRL sec on school computer
                      lambda_function_to_apply_to_pixels=PHI)


num_steering_angles = env.action_space.n
INPUT_SHAPE = env.img_shape


NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 4  # idea: it was taking too long for appearance
# of obstacle to show up in stacked states? switchgin back to sequential memory

#NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 4  # reward_delay * this = prev sec as input
#STACK_EVERY_N_FRAMES = 2

input_shape = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + INPUT_SHAPE

model = Sequential()
model.add(Conv2D(32, kernel_size=3, strides=2,
                 input_shape=input_shape, data_format = 'channels_first'))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=3, strides=2))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=3, strides=2))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(32, activity_regularizer=l2(0.001)))
model.add(Activation('sigmoid'))

model.add(Dense(64, activity_regularizer=l2(0.001)))
model.add(Activation('sigmoid'))

model.add(Dense(num_steering_angles))
print(model.summary())


replay_memory = SequentialMemory(limit=10**4, window_length=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT)
#replay_memory = SkippingMemory(limit=10**4,
#                                              num_states_to_stack=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,
#                                              skip_factor=STACK_EVERY_N_FRAMES)

# something like: w/ probability epsilon (which decays through training),
# select a random action; otherwise, consult the agent
# epsilon = f(x) = ((self.value_max - self.value_min) / self.nb_steps)*x + self.value_max
num_total_training_steps = 150000

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0, # start off 100% random
                              value_min=0.05,  # get to random action x% of time
                              value_test=0.0,  # when testing, take rand action this val *100 % of time
                              nb_steps=int(num_total_training_steps/2)) # of time steps to go from epsilon=value_max to =value_min


ddqn_agent = DQNAgent(model=model, nb_actions=num_steering_angles,
                                  memory=replay_memory, enable_double_dqn=True,
                                  enable_dueling_network=False, target_model_update=1e-1, # soft update parameter?
                                  policy=policy, gamma=0.99, train_interval=4,
                                  nb_steps_warmup=250)

ddqn_agent.compile(Adam(lr=1e-5), metrics=['mae']) # not use mse since |reward| <= 1.0

weights_filename = 'ddqn_collision_avoidance_1201_03.h5'
want_to_train = True
train_from_weights_in_weights_filename = True

if want_to_train is True:

  # note: interval's units are episode_steps
  callbacks_list = [ModelIntervalCheckpoint(filepath=weights_filename, verbose=5, interval=750)]

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
