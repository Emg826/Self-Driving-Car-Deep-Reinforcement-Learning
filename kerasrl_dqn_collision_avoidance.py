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
      "Width": 1024,
      "Height": 512,
      "FOV_Degrees": 120
    },
    {
      "ImageType": 1,
      "Width": 512,
      "Height": 256,
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

#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LocallyConnected2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint  # https://github.com/keras-rl/keras-rl/blob/171667dce2a39993705b12fdf0b3cc3bb7bf26d2/rl/callbacks.py

from airsim_env import AirSimEnv
from multiinput_dqn import MDQNAgent
from skipping_memory import SkippingMemory

np.random.seed(7691)
random.seed(6113)

# This block solved the "CUBLAS_STATUS_ALLOC_FAILED" CUDA issue (https://stackoverflow.com/a/52762075)
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))

# have to be careful not to make PHI too complex, else decr num steps per IRL second
PHI = lambda pixel:  min(2048.0 / (pixel+6.0), 255.0)

env = AirSimEnv(num_steering_angles=5,
                      max_num_steps_in_episode=10**4,
                      fraction_of_top_of_scene_to_drop=0.4,
                      fraction_of_bottom_of_scene_to_drop=0.1,
                      fraction_of_top_of_depth_to_drop=0.4,
                      fraction_of_bottom_of_depth_to_drop=0.35,
                      seconds_pause_between_steps=0.03,  # gives rand num generator time to work (wasn't working b4)
                      seconds_between_collision_in_sim_and_register=1.5,  # note avg 4.12 steps per IRL sec on school computer
                      lambda_function_to_apply_to_depth_pixels=PHI)


num_steering_angles = env.action_space.n

NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 1
STACK_EVERY_N_FRAMES = 1

SCENE_INPUT_SHAPE = env.scene_input_shape + (1,)
DEPTH_INPUT_SHAPE = env.depth_planner_input_shape + (1,)
SENSOR_INPUT_SHAPE =  env.sensor_input_shape + (1,)

print(SCENE_INPUT_SHAPE, DEPTH_INPUT_SHAPE, SENSOR_INPUT_SHAPE)


# BEGIN MODEL
#first input model - height, width, num_channels (gray, so only 1 channel)
scene_nn_input = Input(shape=SCENE_INPUT_SHAPE)
scene_conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(2, 2), data_format='channels_last')(scene_nn_input)
scene_pool_1 = MaxPooling2D(pool_size=(2, 2))(scene_conv_1)
scene_local_1 =  LocallyConnected2D(16, kernel_size=(6, 6), activation='relu', strides=(4, 4))(scene_pool_1)
scene_pool_2 = MaxPooling2D(pool_size=(2, 2))(scene_local_1)
scene_flat = Flatten()(scene_pool_2)

# second input model - for depth images which are also grayscale
depth_nn_input = Input(shape=DEPTH_INPUT_SHAPE)
depth_conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(2, 2), data_format='channels_last')(depth_nn_input)
depth_local_1 =  LocallyConnected2D(16, kernel_size=(6, 6), activation='relu', strides=(5, 5))(depth_conv_1)
depth_pool_1 = MaxPooling2D(pool_size=(2, 2))(depth_local_1)
depth_flat = Flatten()(depth_pool_1)

# third input model - for the numeric sensor data
"""
1-2. GPS (x, y) coordinates of car
3. manhattan distance from end point (x, y)
4. yaw/compass direction of car in radians  # use airsim.to_eularian_angles() to get yaw
5. relative bearing in radians (see relative_bearing.py)
6. current steering angle in [-1.0, 1.0]
x. linear velocity (x, y) # no accurate whatsoever (press ';' in sim to see)
7-8. angular velocity (x, y)
9-10. linear acceleration (x, y)
11-12. angular acceleration (x, y)
13. speed
14-15. absolute difference from current location to destination for x and y each
16-17. (x, y) coordinates of destination
"""
sensor_input = Input(shape=SENSOR_INPUT_SHAPE)  # not much of a 'model', really...
sensor_dense_1 =  Dense(32, activation='linear')(sensor_input)
sensor_output = Flatten()(sensor_dense_1)

merge = concatenate([scene_flat, depth_flat, sensor_output])

# interpretation/combination model
merged_dense_1 = Dense(64, activation='relu')(merge)
merged_dense_2 = Dense(64, activation='relu')(merged_dense_1)
final_output = Dense(num_steering_angles, activation='sigmoid')(merged_dense_2)

model = Model(inputs=[scene_nn_input, depth_nn_input, sensor_input], outputs=final_output)
# summarize layers
print(model.summary())

# plot network graph  - - need graphviz installed
#plot_model(model, to_file='multi_ddqn.png')



#replay_memory = SequentialMemory(limit=10**4, window_length=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT)
replay_memory = SkippingMemory(limit=10**4,
                               num_states_to_stack=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,
                               skip_factor=STACK_EVERY_N_FRAMES)

# something like: w/ probability epsilon (which decays through training),
# select a random action; otherwise, consult the agent
# epsilon = f(x) = ((self.value_max - self.value_min) / self.nb_steps)*x + self.value_max
num_total_training_steps = 150000

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0, # start off 100% random
                              value_min=0.10,  # get to random action x% of time
                              value_test=0.0,  # when testing, take rand action this val *100 % of time
                              nb_steps=int(math.sqrt(num_total_training_steps))) # of time steps to go from epsilon=value_max to =value_min


dqn_agent = MDQNAgent(model=model, nb_actions=num_steering_angles,
                                  memory=replay_memory, enable_double_dqn=True,
                                  enable_dueling_network=False, target_model_update=1e-1, # soft update parameter?
                                  policy=policy, gamma=0.99, train_interval=4,  # gamma of 97 because a lot can change from now til end of car run
                                  nb_steps_warmup=2000)

dqn_agent.compile(Adam(lr=0.0001), metrics=['mae']) # not use mse since |reward| <= 1.0

weights_filename = 'dqn_collision_avoidance_1208_00.h5'
want_to_train = True
train_from_weights_in_weights_filename = True

if want_to_train is True:

  # note: interval's units are episode_steps
  callbacks_list = [ModelIntervalCheckpoint(filepath=weights_filename, verbose=5, interval=400)]

  if train_from_weights_in_weights_filename:
    try:
      dqn_agent.load_weights(weights_filename)
      print('Successfully loaded DQN weights')
    except:
      print('Failed to load DQN weights')

  dqn_agent.fit(env, callbacks=callbacks_list, nb_steps=num_total_training_steps,
                      visualize=False, verbose=2)

  dqn_agent.save_weights(weights_filename)
else: # else want to test
    dqn_agent.load_weights(weights_filename)
    dqn_agent.test(env, nb_episodes=12, visualize=True)
