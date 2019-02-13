import time
from airsim_imitation_learning_data_collector import AirSimILDataCollectorAPI
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, SimpleRNN, MaxPooling2D,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import time

NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 5
NUM_STEERING_ANGLES = 7

def get_scene(data_row):
  scene = data_row[0]
  return cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY).reshape(scene.shape[0], scene.shape[1], 1)

num_steering_angles = 7
time_between_steering_decisions = 0.5
safe_quit = True # save even if quit

api = AirSimILDataCollectorAPI(num_steering_angles=num_steering_angles)
arr_of_steering_angles = api.steering_angles

SCENE_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,)+(128, 128*3, 1)
DEPTH_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,)+(96, 96*3, 1)
MISC_INPUT_SHAPE = (NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,)+(8,1)


# BEGIN MODEL - 
#first input model - height, width, num_channels (gray, so only 1 channel)
scene_nn_input = Input(shape=SCENE_INPUT_SHAPE, name='scene_input')  # strid h, w
scene_conv_1 = ConvLSTM2D(32, kernel_size=(8, 8), strides=(4, 4), data_format='channels_last', return_sequences=True)(scene_nn_input)
scene_1_activation = LeakyReLU()(scene_conv_1)
scene_conv_2 = ConvLSTM2D(64, kernel_size=(4, 4), strides=(2, 2), data_format='channels_last', return_sequences=True)(scene_1_activation)
scene_2_activation = LeakyReLU()(scene_conv_2)
scene_conv_3 = ConvLSTM2D(32, kernel_size=(4, 4), strides=(2, 2), data_format='channels_last', return_sequences=True)(scene_2_activation)
scene_3_activation = LeakyReLU()(scene_conv_3)
out_shape = scene_3_activation._keras_shape  # https://github.com/keras-team/keras/issues/1981#issuecomment-301327235
print(out_shape)
scene_reshaped = Reshape(target_shape=(NUM_FRAMES_TO_STACK_INCLUDING_CURRENT, out_shape[2]*out_shape[3]*out_shape[4]))(scene_3_activation)


merged_simplernn_1 = SimpleRNN(64, activation='tanh', return_sequences=True)(scene_reshaped)
merged_simplernn_2 = SimpleRNN(96, activation='tanh')(merged_simplernn_1)
final_output = Dense(NUM_STEERING_ANGLES, activation='linear')(merged_simplernn_2)

model = Model(inputs=[scene_nn_input], outputs=final_output)
print(model.summary())
model.compile(SGD(lr=0.01, decay=0.0000001,  momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('scene_only_imitation_learning_model_1549643762.h5')


# event loop
api.reset_vehicle()
most_recent_scenes =[None] * NUM_FRAMES_TO_STACK_INCLUDING_CURRENT
api.pause_sim()
ct = 0
while not api.arrived_at_destination():
  data_row = api.get_sim_data()
  scene = get_scene(data_row)
  if ct is 0:
    for idx in range(0, NUM_FRAMES_TO_STACK_INCLUDING_CURRENT):
      most_recent_scenes[idx] = scene
    most_recent_scenes = np.array(most_recent_scenes)
  most_recent_scenes = np.array(np.roll(most_recent_scenes, 1))
  most_recent_scenes[0] = scene
  
  angle_idx_vals = model.predict([[most_recent_scenes]])
  print(angle_idx_vals)
  angle_idx = np.argmax(angle_idx_vals[0])
  api.set_steering_angle(angle_idx)  # -1 since keyboard '1' is 0th idx in arr_of_steering_angles


  # execute
  api.unpause_sim()
  time.sleep(time_between_steering_decisions)
  api.pause_sim()
  ct += 1



