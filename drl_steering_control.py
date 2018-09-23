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


helpful for cams: https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md/#available-cameras
https://github.com/Microsoft/AirSim/blob/master/PythonClient/airsim/types.py
https://github.com/Microsoft/AirSim/blob/master/PythonClient/airsim/client.py

helpful for DRL: https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
"""
import airsim
import numpy as np
import time
import os

### NOTE! - Copy and paste the below to replace in your settings.json file
"""
{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "RpcEnabled": true,
  "EngineSound": false,
  "CameraDefaults": {
    "CaptureSettings": [{
      "Width": 350,
      "Height": 260,
      "FOV_Degrees": 90
    }]
  }
}
"""

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam # usually is pretty fast

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealingPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor, Env
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


NUM_IMGS_TO_REMEMBER = (3,) 
IMG_SHAPE = (1190, 260, 4) # W x H x NUM_CHANNELS
INPUT_SHAPE = NUM_IMGS_TO_REMEMBER + IMG_SHAPE
NUM_ACTIONS = 3



class AirSimClientEnv(Env):
  """
  Child class of the abstract base class called 'Env'. Same as the
  OpenAI Gym environment object (https://gym.openai.com/docs/#environments).
  This is the virtual representation of the environment with which
  the agent will interact (client's environment, more or less). All
  keras-rl agent objects take an environment as a parameter.
  """
  def __init__(self):
    self.client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    self.reward_range = (-np.inf, np.inf)

    # gym things; not sure if need to implement
    self.action_space = None  # maybe do a np.arange(-1.0, 1.000000001, n/2.0)
    self.observation_space = None  

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'
    self.setup_my_cameras(self.client)


  def step(self, action):
    """
    Run 1 timestep of the simulation given some action.

    :param action: action provided by the env???

    :returns:
      -- observations: Agent's observation of current env, so the list of
      ImageResponse objects (stitch together in AirSimProcessor)
      -- reward: reward from previous action
      -- done (boolean): true if episode ended; false otherwise
      -- info: dict of with debugging info
    """
    # puts these in the order should be concatenated (left to r)
    cam_names = [self.left_cam_name,
                 self.forward_cam_name,
                 self.right_cam_name,
                 self.backward_cam_name]
    observation = self.request_all_4_sim_images(cam_names)

    # can sub in a better reward function later
    collision_info = self.client.simGetCollisionInfo()
    if collision_info.has_collided is True:
      reward = -1
    else:
      reward = 1

    done = False
    info = {'collision_info' : collision_info}
    
    return observation, reward, done, info
    
  def reset(self):
    """
    When need to reset the environment, do this.
    """
    self.client.reset()
    
  def render(self):
    pass # n/a for AirSim -- all contained in step()

  def close(self):
    pass # garbage collection if need any

  def setup_my_cameras(self, client):
    """
    Helper function to set the left, right, forward, and back cameras up
    on the vehicle as I've see fit.

    :param client: airsim.CarClient() object that
    already connected to the sim
    
    :returns: nada
    """
    # pitch, roll, yaw ; each is in radians where
    # 15 degrees = 0.261799 (don't ask me why they used radians...)
    # 10 degrees = 0.174533, 60 degrees = 1.0472, and 180 degrees = 3.14159
    # reason for +- 0.70: forward camera FOV is 90 degrees or 1.57 radians;
    # 0.7 is roughly half that and seem to work well, so I'm sticking with it
    # NOTE: these images are reflected over a vertical line: left in the image
    # is actually right in the simulation...should be ok for CNN since it is
    # spatially invariant, but if not, then come back and change these
    self.client.simSetCameraOrientation(self.left_cam_name,
                                        airsim.Vector3r(0.0, 0.0, -0.68))
    self.client.simSetCameraOrientation(self.right_cam_name,
                                   airsim.Vector3r(0.0, 0.0, 0.68))
    self.client.simSetCameraOrientation(self.forward_cam_name,
                                        airsim.Vector3r(0.0, 0.0, 0.0))
    self.client.simSetCameraOrientation(self.backward_cam_name,
                                        airsim.Vector3r(0.0, 0.0, 11.5))
    # tbh: i have no idea why 11.5 works (3.14 should've been ok, but wasn't)

  def request_all_4_sim_images(self, cam_names):
    """
    Helper to get_composite_sim_image. Make a request to the simulation from the
    client for a snapshot from each of the 4 cameras.

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

    return self.client.simGetImages(list_of_img_request_objs)


  
class AirSimProcessor(Processor):
  """
  Processes observations from AirSimClientEnv. In this case,
  create the composite/panoramic image from the list of
  ImageResponse objects
  """
  self.left_cam_name = '2'
  self.right_cam_name = '1'
  self.forward_cam_name = '0'
  self.backward_cam_name = '4

  def process_observation(self, observation):
    """
    What to do with the observation/image from AirSimClientEnv.
    Extract the images from the list of ImageReponses objects 
    """
        # puts these in the order should be concatenated (left to r)
    cam_names = [self.left_cam_name,
                 self.forward_cam_name,
                 self.right_cam_name,
                 self.backward_cam_name]
    
    composite_image = self.make_composite_image_from_responses(observation,
                                                               cam_names)
    
    return composite_image

  def make_composite_image_from_responses(self, sim_img_responses, cam_names):
    """
    Take the list of ImageResponse objects (the observation), each with
    uncompressed 1D RGBA binary string representations of images, convert
    the image to a 2D-4 channel numpy array, and then concatenate all of
    the images into a composite (panoramic-ish) image.

    :param sim_img_responses: list of ImageResponse obj from a call to
    request_all_4_sim_images() in the AirSimEnv class
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

    height = sim_img_responses[0].height
    width = sim_img_responses[0].width
    
    for cam_name, sim_img_response in zip(cam_names, sim_img_responses):
      # get a flat, 1D array of the iamge
      img_1D = np.fromstring(sim_img_response.image_data_uint8, dtype=np.uint8)

      # reshape that into a 2D array then flip upside down
      # (because orignal image is flipped)
      #img_2D_RGBA = np.flipud(img_1D.reshape(sim_img_response.height,
      #                                       sim_img_response.width,
      #                                       4))

      # But! CNN is spatial invariant, meaning it doesn't care if the
      # image is flipped or not, so no need to unflip it
      img_2D_RGBA = img_1D.reshape(height, width, 4)

      
      dict_of_2D_imgs.update({ cam_name : img_2D_RGBA})
      
    # now with all images in 2D, 4 channel form, stitch them together
    # NOTE THESE ARRAY INDEXINGS ARE ASSUMING LEFT-FWD-RIGHT-BACK ordering
    # row, column indexing in []'s
    # int((2*width/5)):: is the right 60% of img; 0 is left 2 is right 3 is rear
    composite_img = np.concatenate([ dict_of_2D_imgs[cam_names[3]][:, int((2*width/5))::],
                                     dict_of_2D_imgs[cam_names[0]][:,int((2*width/5))::],
                                     dict_of_2D_imgs[cam_names[1]],
                                     dict_of_2D_imgs[cam_names[2]][:,0:int((3*width/5))],
                                     dict_of_2D_imgs[cam_names[3]][:,0:int((3*width/5))] ], axis=1)

    # for debugging and getting cameras right
    #airsim.write_png(os.path.normpath('imgs/sim_img'+ str(time.time())+'.png'), composite_img)

    return composite_img    
  

    
def init_modeler(num_actions):
  """
  Initialize the Keras model that will be passed into the DQNAgent object

  :param num_actions: number of possible actions the DQNAgent can take. In
  this file, this is the number of different steering angles the agent can
  choose from.

  :returns: an uncompiled Keras model
  """
  model = Sequential()

  model.add(Convolution2D(128, kernel=32, stride=28, input_shape=INPUT_SHAPE,
                          data_format='channels_last'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=8, strides=8))

  model.add(Flatten())

  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(num_actions))
  model.add(Activation('linear'))

  print('Here\'s that model: \n{}'.format(model.summary))

  return model


def main():
  # setup everything
  model = init_modeler(NUM_ACTIONS)
  memory = SequentialMemory(limt=10**5, window_length=NUM_IMGS_TO_REMEMBER)
  processor = AirSimProcessor()
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0,
                                value_min=0.1, value_test = 0.05, nb_steps=10**9)   
  env = AirSimEnv()
  dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS, policy=policy,
                 memory=memory, processor=processor, nb_steps_warmup=10**3,
                 gamma=0.9, target_model_update=10**3, train_interval=4,
                 delta_clip=1.0)
  dqn.compile(Adam(lr=0.05), metrics=['mse'])

  
  # GO!
  dqn.fit(env, callbacks=None, nb_steps=10**6)


if __name__ == '__main__':
  main()
