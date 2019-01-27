############# nn 25th
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
      "Width": 256,
      "Height": 256,
      "FOV_Degrees": 45
    },
    {
      "ImageType": 1,
      "Width": 256,
      "Height": 256,
      "FOV_Degrees": 45
    }]
  }           
}

"""

import numpy as np
import random
import math

#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, SimpleRNN
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint  # https://github.com/keras-rl/keras-rl/blob/171667dce2a39993705b12fdf0b3cc3bb7bf26d2/rl/callbacks.py

from airsim_env import AirSimEnv
from skipping_memory import SkippingMemory
from multi_input_processor import MultiInputProcessor
from transparent_dqn import TransparentDQNAgent
from coord_conv import CoordinateChannel2D


np.random.seed(7691)
random.seed(6113)

# This block solved the "CUBLAS_STATUS_ALLOC_FAILED" CUDA issue (https://stackoverflow.com/a/52762075)
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))

# have to be careful not to make PHI too complex, else decr num steps per IRL second
PHI = lambda pixel: 255.0 if pixel < 4.3 else 450.0 / pixel

need_channel_dimension = True
concat_coord_conv_layers = True
env = AirSimEnv(num_steering_angles=5,
                      max_num_steps_in_episode=650,
                      fraction_of_top_of_scene_to_drop=0.05,
                      fraction_of_bottom_of_scene_to_drop=0.1,
                      fraction_of_top_of_depth_to_drop=0.3,
                      fraction_of_bottom_of_depth_to_drop=0.45,
                      seconds_pause_between_steps=0.2,  # assuming sim clock =1.0, 1/this is num steps per sim sec
                      seconds_between_collision_in_sim_and_register=0.4,  # note avg 4.12 steps per IRL sec on school computer
                      lambda_function_to_apply_to_depth_pixels=PHI,
                      need_channel_dimension=need_channel_dimension)  # NN doesn't care if image looks  nice
                      # leaving ^ as None almost doubles num steps per IRL second, meaning
                      # can increase sim speed an get more done!


num_steering_angles = env.action_space.n

NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 1
STACK_EVERY_N_FRAMES = 1  # don't change this for now

SCENE_INPUT_SHAPE = None
DEPTH_INPUT_SHAPE = None
SENSOR_INPUT_SHAPE = None
# split into if else because want to avoid the extra
# (...,1) at the end messes up the dimensionality of input to NN
# when the inputs are stacked
if NUM_FRAMES_TO_STACK_INCLUDING_CURRENT == 1:
  # this works whether doing CoordConv+Conv2D or whether doign regular Conv2D
  SCENE_INPUT_SHAPE = env.SCENE_INPUT_SHAPE + (1,)
  DEPTH_INPUT_SHAPE = env.DEPTH_PLANNER_INPUT_SHAPE + (1,)
  SENSOR_INPUT_SHAPE =  env.SENSOR_INPUT_SHAPE + (1,)

else:
  # 5th dimension would be the channel dimension; don't need this if Conv2D
  # but do need this if ConvRNN2D or ConvRNN
  if need_channel_dimension:
    SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE + (1,)
    DEPTH_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.DEPTH_PLANNER_INPUT_SHAPE + (1,)
    SENSOR_INPUT_SHAPE =  (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SENSOR_INPUT_SHAPE + (1,)

  else:
    SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE
    DEPTH_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.DEPTH_PLANNER_INPUT_SHAPE
    SENSOR_INPUT_SHAPE =  (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SENSOR_INPUT_SHAPE

print(SCENE_INPUT_SHAPE, DEPTH_INPUT_SHAPE, SENSOR_INPUT_SHAPE)


# BEGIN MODEL - 
#first input model - height, width, num_channels (gray, so only 1 channel)  # note - stride is what reduces the dimensions
scene_nn_input = Input(shape=SCENE_INPUT_SHAPE)
scene_coord_conv_inputs = CoordinateChannel2D(use_radius=True, data_format='channels_last')(scene_nn_input)

scene_conv_1 = Conv2D(64, kernel_size=(8,8), strides=(3, 4), data_format='channels_last')(scene_coord_conv_inputs)
scene_1_activation = LeakyReLU(0.1)(scene_conv_1)

scene_conv_2 = Conv2D(128, kernel_size=(5,5), strides=(2, 3), data_format='channels_last')(scene_1_activation)
scene_2_activation = LeakyReLU(0.1)(scene_conv_2)

scene_conv_3 = Conv2D(128, kernel_size=(4,4), strides=(2, 3), data_format='channels_last')(scene_2_activation)
scene_3_activation = LeakyReLU(0.1)(scene_conv_3)

scene_conv_4 = Conv2D(96, kernel_size=(4,3), strides=(2, 2), data_format='channels_last')(scene_3_activation)
scene_4_activation = LeakyReLU(0.1)(scene_conv_4)

scene_flat = Flatten()(scene_4_activation)


# not as deep as scene NN because depth not contain as much info per image
depth_nn_input = Input(shape=DEPTH_INPUT_SHAPE)
depth_coord_conv_input = CoordinateChannel2D(use_radius=True, data_format='channels_last')(depth_nn_input)

depth_conv_1 = Conv2D(64, kernel_size=(8,8), strides=(3, 4), data_format='channels_last')(depth_coord_conv_input)
depth_1_activation = LeakyReLU(0.1)(depth_conv_1)

depth_conv_2 = Conv2D(96, kernel_size=(5,5), strides=(2, 3), data_format='channels_last')(depth_1_activation)
depth_2_activation = LeakyReLU(0.1)(depth_conv_2)

depth_conv_3 = Conv2D(64, kernel_size=(4,4), strides=(2, 4), data_format='channels_last')(depth_2_activation)
depth_3_activation = LeakyReLU(0.1)(depth_conv_3)

depth_flat = Flatten()(depth_3_activation)

# third input model - for the numeric sensor data  # 218 and 64 x 768
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
# SENSOR_INPUT_SHAPE[0] * SENSOR_INPUT_SHAPE[0]
sensor_output = Flatten()(sensor_input)

merge = concatenate([scene_flat, depth_flat, sensor_output])

# interpretation/combination model
# fully connected layers
merged_dense_1 = Dense(512, activation='sigmoid')(merge)
merged_dense_2 = Dense(1024, activation='sigmoid')(merged_dense_1)
final_output = Dense(num_steering_angles, activation='linear')(merged_dense_2)


""" # ISSUE: RNN really only makes sense if there is a sequence of inputs;
# however, CNN's filters will scan all N stacked frames @ once: ie it will
# mix the sequence together. 
merged_simplernn_1 = SimpleRNN(512, activation='tanh')(merge)
merged_simplernn_2 = SimpleRNN(1024, activation='tanh')(merged_simplernn_1)
final_output = Dense(num_steering_angles, activation='linear')(merged_simplernn_2)
"""

model = Model(inputs=[scene_nn_input, depth_nn_input, sensor_input], outputs=final_output)
# summarize layers
print(model.summary())

# plot network graph  - - need graphviz installed
#plot_model(model, to_file='multi_ddqn.png')



#replay_memory = SequentialMemory(limit=10**4, window_length=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT)
replay_memory = SkippingMemory(limit=10000,
                               num_states_to_stack=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,
                               skip_factor=STACK_EVERY_N_FRAMES)

# something like: w/ probability epsilon (which decays through training),
# select a random action; otherwise, consult the agent
# epsilon = f(x) = ((self.value_max - self.value_min) / self.nb_steps)*x + self.value_max


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0, # start off 100% random
                              value_min=0.10,  # get to random action x% of time
                              value_test=0.01,  # MUST BE >0 else, for whatever reason, won't get random start
                              nb_steps=100000) # of time steps to go from epsilon=value_max to =value_min


multi_input_processor = MultiInputProcessor(num_inputs=3, num_inputs_stacked=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT) # 3 inputs: scene img, depth img, sensor data

# compute gamma -  a lot can change from now til end of car run -
future_time_steps_until_discount_rate_is_one_half = 25.0  # assuming ~ 4 time steps per simulation second
# solve gamma ^ n = 0.5 for some n - kind of like a half life?
discount_rate = math.exp( math.log(0.5, math.e) / future_time_steps_until_discount_rate_is_one_half )

train_every_n_steps = 4
dqn_agent = TransparentDQNAgent(model=model,nb_actions=num_steering_angles,
                                  memory=replay_memory, enable_double_dqn=True,
                                  enable_dueling_network=False, target_model_update=10000, # was soft update parameter?
                                  policy=policy, gamma=discount_rate, train_interval=train_every_n_steps,     
                                  nb_steps_warmup=512, batch_size=16,   # i'm going to view gamma like a confidence level in q val estimate
                                  processor=multi_input_processor,
                                  print_frequency=5)

#https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L157:
# lr := lr * ( 1 / (1 + (decay * iterations)))
dqn_agent.compile(SGD(lr=0.005, decay=0.001665), metrics=['mae']) # not use mse since |reward| <= 1.0

weights_filename = 'dqn_collision_avoidance_012619_03.h5'
want_to_train = True
load_in_weights_in_weights_filename = True
num_total_training_steps = 1000000
if want_to_train is True:

  # note: interval's units are episode_steps
  callbacks_list = [ModelIntervalCheckpoint(filepath=weights_filename, verbose=5, interval=400)]

  if load_in_weights_in_weights_filename:
    try:
      dqn_agent.load_weights(weights_filename)
      print('Successfully loaded DQN weights')
    except:
      print('Failed to load DQN weights')

  dqn_agent.fit(env, callbacks=callbacks_list, nb_steps=num_total_training_steps,
                      visualize=False, verbose=False)
  
  if env.total_num_steps > 5000:
    dqn_agent.memory.write_transitions_to_file()
    
  dqn_agent.save_weights(weights_filename)
else: # else want to test
    dqn_agent.load_weights(weights_filename)
    dqn_agent.test(env, nb_episodes=10, visualize=False)
