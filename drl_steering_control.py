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


import airsim
import numpy as np
import time
import os

def get_composite_sim_image(client):
  """
  Get snapshots from the left, forward, right, and rear cameras in 1 2D numpy array
  
  :param:client: airsim.CarClient() object that has already connected
  to the simulation

  :returns: 2D numpy array that is a composite of all 4 images. 

  a little help from: https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md/#Available%20Cameras
  """

  # puts these in the order in which the images thereof should be concatenated (left to r)
  cam_names = ['front_left', 'front_center', 'front_right', 'back_center']

  # make the request for a snapshot from all 4 cameras (except the first person camera)
  sim_img_responses = request_all_4_sim_images(client, cam_names)
  
  return make_composite_image_from_responses(sim_img_responses, cam_names)


def request_all_4_sim_images(client, cam_names):
  """
  Make a request to the simulation from the client for a snapshot from each of
  the 4 cameras.

  :param client: airsim.CarClient() object that has already connected to the simulation
  :param cam_names: list of camera names, order of list is order in which requests will
  be made and (presumably) returned
  
  :returns: list where each element is an airsim.ImageResponse() obj
  """
  # build list of airsim.ImageRequest objects to give to client.simGetImages()
  list_of_img_request_objs = []
  for cam_name in cam_names:
    list_of_img_request_objs.append( airsim.ImageRequest(camera_name=cam_name,
                                                               image_type=airsim.ImageType.Scene,
                                                               pixels_as_float=False,
                                                               compress=False) )

  return client.simGetImages(list_of_img_request_objs)


def make_composite_image_from_responses(sim_img_responses, cam_names):
  """
  Take the lists responses, each with uncompressed 1D RGBA binary string
  representations of images, convert the image to a 2D-4 channel numpy array,
  and then concatenate all of the images into a composite image.

  :param sim_img_responses: dictionary of the responses from a call to
  request_all_4_sim_images()
  :param cam_names: names of the cameras from which the image request objs
  were gotten; order of this names in cam_names is ASSUMED to be the same
  order in sim_img_responses
  
  :returns: 2D, 4 channel numpy array of all images "stitched" together as:
  left forward right back  (NOTE: not sure where to put back img).

  Example numpy array: [ [(100, 125, 150, 1) , (255, 100, 255, 1)],
                         [(255, 255, 255, 1) , (255, 255, 0, 1)] ]
  """
  # extract the 1D  RGBA binary uint8 string images into 2D,
  # 4 channel images; append that image to a list

  # order of these names is order images will be concatenated
  # together (from left to right)
  
  dict_of_2D_imgs = {}

  for cam_name, sim_img_response in zip(cam_names, sim_img_responses):
    # get a flat, 1D array of the iamge
    img_1D = np.fromstring(sim_img_response.image_data_uint8, dtype=np.uint8)

    # reshape that into a 2D array then flip upside down
    # (because orignal image is flipped)
    img_2D_RGBA = np.flipud(img_1D.reshape(sim_img_response.height,
                                           sim_img_response.width,
                                           4))

    
    dict_of_2D_imgs.update({ cam_name : img_2D_RGBA})
    
  print(len(sim_img_responses))
  # now with all images in 2D, 4 channel form, stitch them together
  composite_img = np.concatenate([ dict_of_2D_imgs[cam_names[0]],
                                   dict_of_2D_imgs[cam_names[1]],
                                   dict_of_2D_imgs[cam_names[2]],
                                   dict_of_2D_imgs[cam_names[3]] ], axis=1)

  # for debugging
  airsim.write_png(os.path.normpath('sim_img'+ str(time.time())+'.png'), composite_img)

  return composite_img    
    


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
  composite_img = get_composite_sim_image(client)
  print(composite_img.shape)
  time.sleep(2)

  time_step += 1
  # request an image of the scene from the front facing camera


    # assume i have all 4 images. Now what? do i concatenate the images?
    # maybe do this: https://keras.io/layers/merge/#concatenate

    # when ^ is figured out, look @ https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    # for an example of how to use keras-rl
