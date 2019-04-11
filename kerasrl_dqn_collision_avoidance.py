############# nn 25th
want_to_train = False

"""https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py"""
"""https://github.com/Kjell-K/AirGym/blob/master/DQN-Train.py"""
"""https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras"""

""" Copy and paste one of these into your settings.json (which hould be in your Documents folder)
{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "RpcEnabled": true,
  "ViewMode": "NoDisplay",
  "EngineSound": false,
  "ClockSpeed": 1.0,
  "CameraDefaults": {
    "CaptureSettings": [{
      "ImageType": 0,
      "Width": 128,
      "Height": 128,
      "FOV_Degrees": 90
    },
    {
      "ImageType": 1,
      "Width": 96,
      "Height": 96,
      "FOV_Degrees": 90
    }]
  }           
}



OR

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
      "Width": 128,
      "Height": 128,
      "FOV_Degrees": 90
    },
    {
      "ImageType": 1,
      "Width": 96,
      "Height": 96,
      "FOV_Degrees": 90
    }]
  }           
}
"""

import numpy as np
import random
import math

#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, SimpleRNN, MaxPooling2D,Reshape
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
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


# This block solved the "CUBLAS_STATUS_ALLOC_FAILED" CUDA issue (https://stackoverflow.com/a/52762075)
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))

# have to be careful not to make PHI too complex, else decr num steps per IRL second
PHI = lambda pixel: 1.0 / (pixel+1.0)

random.seed(123)

# IRL seconds # x1.0 speed: 9.4 steps per sim sec @ 0.05 wait; 7.5 spss @ 0.1; 3.7 spss @ 0.2 wait
# IRL seconds  #x2.0 speed: 4.5 steps per IRL second @ 0,2 wait
need_channel_dimension = True
concat_x_y_channels = True
scene_in_grayscale = False
want_depth_image = False
train_every_n_steps = 4
env = AirSimEnv(num_steering_angles=5,  # should be an odd number so as to include 0
                      max_num_steps_in_episode=1000,
                      fraction_of_top_of_scene_to_drop=0.0,  # leaving these as 0 so as to keep square images
                      fraction_of_bottom_of_scene_to_drop=0.0,
                      fraction_of_top_of_depth_to_drop=0.0,
                      fraction_of_bottom_of_depth_to_drop=0.0,
                      seconds_pause_between_steps=0.5,  # it's not a linear scaling down: if go from x1 speed w/ wait 0.2, cant do 0.1 wait for x2 speed
                      seconds_between_collision_in_sim_and_register=0.2,  # note avg 4.12 steps per IRL sec on school computer
                      lambda_function_to_apply_to_depth_pixels=PHI,
                      need_channel_dimension=need_channel_dimension,
                      depth_settings_md_size=(96,96),
                      scene_settings_md_size=(128,128),
                      proximity_instead_of_depth_planner=False,
                      concat_x_y_coords_to_channel_dim=concat_x_y_channels,
                      convert_scene_to_grayscale=scene_in_grayscale,
                      want_depth_image=want_depth_image,
                      train_frequency=train_every_n_steps)  # NN doesn't care if image looks  nice
                      # leaving ^ as None almost doubles num steps per IRL second, meaning
                      # can increase sim speed an get more done!


num_steering_angles = env.action_space.n

NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 3
STACK_EVERY_N_FRAMES = 1  # don't change this for now

SCENE_INPUT_SHAPE = None
DEPTH_INPUT_SHAPE = None
SENSOR_INPUT_SHAPE = None
PROXIMITY_INPUT_SHAPE = None
# split into if else because want to avoid the extra
# (...,1) at the end messes up the dimensionality of input to NN
# when the inputs are stacked
if NUM_FRAMES_TO_STACK_INCLUDING_CURRENT == 1:
  # this works whether doing CoordConv+Conv2D or whether doign regular Conv2D
  SCENE_INPUT_SHAPE = env.SCENE_INPUT_SHAPE + (1,)
  DEPTH_INPUT_SHAPE = env.DEPTH_PLANNER_INPUT_SHAPE + (1,)
  SENSOR_INPUT_SHAPE =  env.SENSOR_INPUT_SHAPE + (1,)
  PROXIMITY_INPUT_SHAPE = env.PROXIMITY_INPUT_SHAPE + (1,)

else:
  # 5th dimension would be the channel dimension; don't need this if Conv2D
  # but do need this if ConvRNN2D or ConvRNN
  if need_channel_dimension:
    if concat_x_y_channels is True:
      if scene_in_grayscale:
        SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE + (1+2,)
      else:
        SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE + (3+2,)
        
      DEPTH_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.DEPTH_PLANNER_INPUT_SHAPE + (3,)
      SENSOR_INPUT_SHAPE =  (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SENSOR_INPUT_SHAPE + (1,)
      PROXIMITY_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.PROXIMITY_INPUT_SHAPE + (1,)
    else:
      if scene_in_grayscale:
        SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE + (1,)
      else:
        SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE + (3,)
        
      DEPTH_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.DEPTH_PLANNER_INPUT_SHAPE + (1,)
      SENSOR_INPUT_SHAPE =  (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SENSOR_INPUT_SHAPE + (1,)
      PROXIMITY_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.PROXIMITY_INPUT_SHAPE + (1,)


  else:
    if scene_in_grayscale:
      SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE
    else:
      SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SCENE_INPUT_SHAPE + (3,)

    DEPTH_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.DEPTH_PLANNER_INPUT_SHAPE
    SENSOR_INPUT_SHAPE =  (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.SENSOR_INPUT_SHAPE
    PROXIMITY_INPUT_SHAPE =  (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,) + env.PROXIMITY_INPUT_SHAPE


print(SCENE_INPUT_SHAPE, DEPTH_INPUT_SHAPE, SENSOR_INPUT_SHAPE, PROXIMITY_INPUT_SHAPE)




# BEGIN MODEL - 
#first input model - height, width, num_channels (gray, so only 1 channel)
scene_nn_input = Input(shape=SCENE_INPUT_SHAPE)  # strid h, w
scene_conv_1 = Conv3D(64, kernel_size=(1, 8, 8), strides=(1, 2, 2), data_format='channels_last')(scene_nn_input)
scene_1_activation = LeakyReLU()(scene_conv_1)

scene_conv_2 = Conv3D(64, kernel_size=(1, 4, 4), strides=(1, 2, 2), data_format='channels_last')(scene_1_activation)
scene_2_activation = LeakyReLU()(scene_conv_2)

scene_conv_3 = Conv3D(96, kernel_size=(1, 4, 4), strides=(1, 2, 2), data_format='channels_last')(scene_2_activation)
scene_3_activation = LeakyReLU()(scene_conv_3)

# new as of 0227 923am
scene_conv_4 = Conv3D(64, kernel_size=(1, 2, 2), strides=(1, 1, 1), data_format='channels_last')(scene_3_activation)
scene_4_activation = LeakyReLU()(scene_conv_4)

#scene_flat = Flatten()(scene_4_activation)
print(scene_4_activation._keras_shape)
out_shape = scene_4_activation._keras_shape  # https://github.com/keras-team/keras/issues/1981#issuecomment-301327235
scene_reshaped = Reshape(target_shape=(NUM_FRAMES_TO_STACK_INCLUDING_CURRENT, out_shape[2]*out_shape[3]*out_shape[4]))(scene_4_activation)


"""
# not as deep as scene NN because depth not contain as much info per image
depth_nn_input = Input(shape=DEPTH_INPUT_SHAPE)

depth_conv_1 = Conv3D(16, kernel_size=(1, 8 ,8), strides=(1, 4,4), data_format='channels_last')(depth_nn_input)
depth_1_activation = LeakyReLU()(depth_conv_1)

depth_conv_2 = Conv3D(32, kernel_size=(1, 6, 6), strides=(1, 3, 3), data_format='channels_last')(depth_1_activation)
depth_2_activation = LeakyReLU()(depth_conv_2)

#depth_flat = Flatten()(depth_3_activation)
out_shape = depth_2_activation._keras_shape  # want to flatten the convolution, but keep the temporal stack
depth_reshaped = Reshape(target_shape=(NUM_FRAMES_TO_STACK_INCLUDING_CURRENT, out_shape[2]*out_shape[3]*out_shape[4]))(depth_2_activation)
"""

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
print(sensor_input._keras_shape)
sensor_reshaped = Reshape(target_shape=(sensor_input._keras_shape[1], sensor_input._keras_shape[2]))(sensor_input)
# SENSOR_INPUT_SHAPE[0] * SENSOR_INPUT_SHAPE[0]
#sensor_output = Flatten()(sensor_input)

#merge = concatenate([scene_reshaped, depth_reshaped, sensor_reshaped])
merge = concatenate([scene_reshaped, sensor_reshaped])
merge = Reshape(target_shape=( merge._keras_shape[1], merge._keras_shape[2]) )(merge)

print(merge._keras_shape)



# ISSUE: RNN really only makes sense if there is a sequence of inputs;
# however, CNN's filters will scan all N stacked frames @ once: ie it will
# mix the sequence together.
"""# 02132019_02
merged_simplernn_1 = SimpleRNN(128, activation='tanh', return_sequences=True)(merge)
merged_simplernn_2 = SimpleRNN(256, activation='tanh')(merged_simplernn_1)
final_output = Dense(num_steering_angles, activation='linear')(merged_simplernn_2)


merged_simplernn_1 = SimpleRNN(256, activation='tanh', return_sequences=True)(merge)
merged_simplernn_2 = SimpleRNN(512, activation='tanh', return_sequences=True)(merged_simplernn_1)
final_output = SimpleRNN(num_steering_angles, activation='linear')(merged_simplernn_2)


"""
merged_simplernn_1 = SimpleRNN(192, activation='sigmoid', return_sequences=True)(merge)
merged_simplernn_2 = SimpleRNN(256, activation='sigmoid', return_sequences=False)(merged_simplernn_1)
final_output = Dense(num_steering_angles, activation='linear')(merged_simplernn_2)

#model = Model(inputs=[scene_nn_input, depth_nn_input, sensor_input], outputs=final_output)
model = Model(inputs=[scene_nn_input, sensor_input], outputs=final_output)

# summarize layers
print(model.summary())


# plot network graph  - - need graphviz installed
#plot_model(model, to_file='multi_ddqn.png')


replay_memory = SequentialMemory(limit=3000, window_length=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT)
#replay_memory = SkippingMemory(limit=8000,
#                               num_states_to_stack=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,
#                               skip_factor=STACK_EVERY_N_FRAMES)

# something like: w/ probability epsilon (which decays through training),
# select a random action; otherwise, consult the agent
# epsilon = f(x) = ((self.value_max - self.value_min) / self.nb_steps)*x + self.value_max

num_total_training_steps = 5000
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                        attr='eps',
                                        value_max=0.75, # start off 100% random
                                        value_min=0.05,  # get to random action x% of time
                                        value_test=0.00001,  # MUST BE >0 else, for whatever reason, won't get random start
                                        nb_steps=num_total_training_steps) # of time steps to go from epsilon=value_max to =value_min


multi_input_processor = MultiInputProcessor(num_inputs=2+want_depth_image, num_inputs_stacked=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT) # 3 inputs: scene img, depth img, sensor data

# compute gamma -  a lot can change from now til end of car run -
future_time_steps_until_discount_rate_is_one_half = 8.0  # assuming ~ 4 time steps per simulation second
# solve gamma ^ n = 0.5 for some n - kind of like a half life?
discount_rate = math.exp( math.log(0.5, math.e) / future_time_steps_until_discount_rate_is_one_half )

train_every_n_steps = 3
dqn_agent = TransparentDQNAgent(model=model,nb_actions=num_steering_angles,
                                  memory=replay_memory, enable_double_dqn=True,
                                  enable_dueling_network=False, target_model_update=2500, # was soft update parameter?
                                  policy=policy, gamma=discount_rate, train_interval=train_every_n_steps,     
                                  nb_steps_warmup=64, batch_size=5,   # i'm going to view gamma like a confidence level in q val estimate
                                  processor=multi_input_processor,
                                  print_frequency=12)

#https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L157:
#https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# lr := lr * ( 1 / (1 + (decay * iterations)))
init_lr = 1e-2  # lr was too high and caused weights to go to NaN --> output NaN for Q-values
lr_decay_factor = init_lr / (float(num_total_training_steps) / train_every_n_steps) # lr / (1. + lr_factor_decay) each train step
dqn_agent.compile(SGD(lr=init_lr, decay=lr_decay_factor), metrics=['mae']) # not use mse since |reward| <= 1.0

weights_filename = 'dqn_collision_avoidance_03312019_sometimes_works - Copy.h5'
#weights_filename = 'dqn_collision_avoidance_012619_03_coordconv_circleTheIntersection.h5'
load_in_weights_in_weights_filename = True
if want_to_train is True:

  # note: interval's units are episode_steps
  callbacks_list = [ModelIntervalCheckpoint(filepath=weights_filename, verbose=5, interval=600)]

  if load_in_weights_in_weights_filename:
    try:
      dqn_agent.load_weights(weights_filename)
      print('Successfully loaded DQN weights')
    except:
      print('Failed to load DQN weights')

  dqn_agent.fit(env, callbacks=callbacks_list, nb_steps=num_total_training_steps,
                      visualize=False, verbose=False)
  
  if env.total_num_steps > 20000:
    #dqn_agent.memory.write_transitions_to_file()
    2==2
    
  dqn_agent.save_weights(weights_filename, overwrite=True)
else: # else want to test
    dqn_agent.load_weights(weights_filename)
    dqn_agent.test(env, nb_episodes=200, visualize=False)
