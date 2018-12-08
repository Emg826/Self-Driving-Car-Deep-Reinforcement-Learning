# based on:
# https://machinelearningmastery.com/keras-functional-api-deep-learning/  # section 5

import numpy as np

# want to put x_images and x_sensor_data into a singele neural network.
"""
x_sensor_data     x_images
      \             /
       \           /
        \         /
         \      ----- flatten
         --------- merge/concatenate the 2
             |   fully conn'd
             .
             .
             .
             | output layer

"""


# Multiple Inputs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LocallyConnected2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

SCENE_INPUT_SHAPE = (512, 1024, 1)
DEPTH_INPUT_SHAPE = (100, 512, 1)
INFRARED_INPUT_SHAPE = (64, 128, 1)
SENSOR_INPUT_SHAPE = (27,)

# height, width, num_channels (gray, so only 1 channel)
scene_nn_input = Input(shape=SCENE_INPUT_SHAPE)
scene_conv_1 = Conv2D(32, kernel_size=(12, 12), activation='relu', strides=(8, 8))(scene_nn_input)
scene_pool_1 = MaxPooling2D(pool_size=(2, 2))(scene_conv_1)
scene_local_1 =  LocallyConnected2D(32, kernel_size=(16, 16), activation='relu', strides=(12, 12))(scene_pool_1)
scene_pool_2 = MaxPooling2D(pool_size=(2, 2))(scene_local_1)
scene_flat = Flatten()(scene_pool_2)

# second input model - for depth images which are also grayscale
depth_nn_input = Input(shape=DEPTH_INPUT_SHAPE)
depth_conv_1 = Conv2D(32, kernel_size=(5, 5), activation='relu', strides=(3, 3))(depth_nn_input)
depth_pool_1 = MaxPooling2D(pool_size=(2, 2))(depth_conv_1)
depth_local_1 =  LocallyConnected2D(32, kernel_size=(8, 8), activation='relu', strides=(7, 7))(depth_pool_1)
depth_pool_1 = MaxPooling2D(pool_size=(2, 2))(depth_local_1)
depth_flat = Flatten()(depth_pool_1)

# third input model - for low-res infrared images which are also grayscale
infrared_nn_input = Input(shape=INFRARED_INPUT_SHAPE)
infrared_conv_1 =  Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(3, 3))(infrared_nn_input)
infrared_pool_1 = MaxPooling2D(pool_size=(2, 2))(infrared_conv_1)
infrared_local_1 = LocallyConnected2D(8, kernel_size=(4, 4), activation='relu', strides=(3, 3))(infrared_pool_1)
infrared_pool_2 = MaxPooling2D(pool_size=(2, 2))(infrared_local_1)
infrared_flat = Flatten()(infrared_pool_2)

# fourth input model - for the numeric sensor data
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
sensor_output =  Dense(32, activation='linear')(sensor_input)

merge = concatenate([scene_flat, depth_flat, infrared_flat, sensor_output])

# interpretation/combination model
merged_dense_1 = Dense(64, activation='relu')(merge)
merged_dense_2 = Dense(64, activation='relu')(merged_dense_1)
final_output = Dense(1, activation='sigmoid')(merged_dense_2)

model = Model(inputs=[scene_nn_input, depth_nn_input, infrared_nn_input, sensor_input], outputs=final_output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png')


# garbage data to test if the architecture will even work - scene, depth, infra, sens
x1 = [np.random.random_sample( SCENE_INPUT_SHAPE ) for _ in range(0, 100)]
x2 = [np.random.random_sample( DEPTH_INPUT_SHAPE) for _ in range(0, 100)]
x3 = [np.random.random_sample( INFRARED_INPUT_SHAPE) for _ in range(0, 100)]
x4 = [np.random.random_sample( SENSOR_INPUT_SHAPE) for _ in range(0, 100)]

y =  [np.random.randint(0, 1+1) for _ in range(0, 100)]

model.compile(optimizer='adam', loss='binary_crossentropy')
plot_model(model, to_file='multiple_inputs.png')


model.fit([x1, x2, x3, x4], y, epochs=20)

# it works !!!!!!!! :)


"""
# assuming i didn't mess anything up while copy pasting, this below should work
# https://machinelearningmastery.com/keras-functional-api-deep-learning/  # section 5
# first input model - for regular images grayscale HD camera images
visible1 = Input(shape=(64,64,1))

conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)

# second input model
visible2 = Input(shape=(32,32,3))
conv21 = Conv2D(32, kernel_size=4, activation='relu')(visible2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png')


# my contrib
x1 = [np.random.rand( 64, 64, 1) for _ in range(0, 100)]
x2 = [np.random.rand( 32, 32, 3) for _ in range(0, 100)]
y =  [np.random.randint(0, 1+1) for _ in range(0, 100)]

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit([x1, x2], y, epochs=10)
"""
