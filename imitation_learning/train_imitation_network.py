#from keras.utils import plot_model
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


# load the data - assumes already preprocessed by raw_....py
x_scenes = np.load('npy/x_scenes_1549639843.npy')
x_depths = np.load('npy/x_depths_1549639843.npy')
x_miscs = np.load('npy/x_miscs_1549639843.npy')
y = np.load('npy/y_1549639843.npy')


NUM_FRAMES_TO_STACK_INCLUDING_CURRENT = 5
NUM_STEERING_ANGLES = 7

generador = TimeseriesGenerator(x_scenes, y,
                                length=NUM_FRAMES_TO_STACK_INCLUDING_CURRENT,
                                batch_size=1,
                                start_index=0, end_index=x_scenes.shape[0]-NUM_FRAMES_TO_STACK_INCLUDING_CURRENT-1)

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

init_lr = 1e-3  # lr 1e-2 for 5 epochs  - 1e-3 w/ added momentum=.9
lr_decay_factor = 1e-12
model.compile(SGD(lr=init_lr, decay=lr_decay_factor,  momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('scene_only_imitation_learning_model_1549643762.h5')
model.fit_generator(generador, epochs=5, verbose=1)

model.save('scene_only_imitation_learning_model_{}.h5'.format(int(time.time())))




