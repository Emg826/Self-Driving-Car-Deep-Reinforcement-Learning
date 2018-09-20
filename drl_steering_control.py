"""
The purpose of this file is to create a deep reinforcement learning
system to control the steering ONLY. The throttle will be applied such
that the maintained speed is 25 mph.

Therefore, the first goal is to get a deep reinforcement algorithm that can,
at the very least, steer itself to avoid collisions. If the algorithm is
able to accomplish this fairly quickly, lane keeping should be next.

The second goal is to come up with a usable reward/penalty function. The
intended function is (distance_since_last_infraction)^2 where infractions
are things such as collisions and resets. Note: I'm not sure how to incorporate
penalties (like hitting a pedestrian).

The third and last goal (for this file at least) is to just figure out how to
use multiple simulation images with a deep Q network. This means learning how
use multiple images and how to actually implement the deep Q network.

The hope is that this will help to create a reward function and deep network
that could be taken outside the simulation and continue learning. Therefore,
I have to figure out how to take advantage of the simulation (e.g., it
tracks collisions) and how to keep it as realistic as possible.
"""


## COPY THIS INSIDE THE OUTER MOST {} OF THE SETTING.JSON FILE FOR THE SIM
# (in your Documents/AirSim folder if you're on windows)

"""
  "Vehicles": {
    "PhysXCar": {
      "Cameras": {
        "front": {
          "CaptureSettings": [{
              "ImageType": 0,
              "Width": 512,
              "Height" 256
          }],
          "Yaw": 0,
        },
        "right": {
          "CaptureSettings": [{
              "ImageType": 0,
              "Width": 512,
              "Height" 256
          }],
          "Yaw": -90,
        },
        "left": {
          "CaptureSettings": [{
              "ImageType": 0,
              "Width": 512,
              "Height" 256
          }],
          "Yaw": 90,
        },
        "rear": {
          "CaptureSettings": [{
              "ImageType": 0,
              "Width": 512,
              "Height" 256
          }],
          "Yaw": 180,
        },

      }
    }
  },
"""


import airsim
import numpy as np
import time



# client init
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# car controls struct init
car_controls = airsim.CarControls()

# collision info struct init
collision_info = client.simGetCollisionInfo()

#
#
# <deep reinforcment learning algorithm here>
#
#

time_step = 0

# simulation/client event loop
while True:
  print('Round {}'.format(time_step+1))
  # request an image of the scene from the front facing camera
  sim_img_responses = client.simGetImages([airsim.ImageRequest(camera_name='left',
                                                               image_type=airsim.ImageType.Scene,
                                                               pixels_as_float=True,
                                                               compress=False),
                                           airsim.ImageRequest(camera_name='front',
                                                               image_type=airsim.ImageType.Scene,
                                                               pixels_as_float=True,
                                                               compress=False),
                                           airsim.ImageRequest(camera_name='right',
                                                               image_type=airsim.ImageType.Scene,
                                                               pixels_as_float=True,
                                                               compress=False),
                                           airsim.ImageRequest(camera_name='rear',
                                                               image_type=airsim.ImageType.Scene,
                                                               pixels_as_float=True,
                                                               compress=False)])

    left_img = sim_img_responses['left'].image_data_float
    front_img = sim_img_responses['front'].image_data_float
    right_img = sim_img_responses['right'].image_data_float
    rear_img = sim_img_responses['rear'].image_data_float

    # assume i have all 4 images. Now what? do i concatenate the images?
    # maybe do this: https://keras.io/layers/merge/#concatenate

    # when ^ is figured out, look @ https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    # for an example of how to use keras-rl
