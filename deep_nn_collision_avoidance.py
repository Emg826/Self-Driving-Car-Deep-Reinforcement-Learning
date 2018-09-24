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

import airsim
import numpy as np
import time
import os
import random

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam # usually is pretty fast

IMG_SHAPE = (1190, 260, 4) # W x H x NUM_CHANNELS
random.seed(3)

class AirSimEnv():
  """
  This is the virtual representation of the environment with which
  the agent will interact. To the driver, abstraction allows the
  "environment" of the simulation to just be represented by an
  image and the car.
  """
  def __init__(self):
    self.client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'
    
    self.setup_my_cameras(self.client)

  def get_environment_state(self):
    """
    Get state of the environment and the vehicle in the environment.

    :returns: panoramic image, numpy array with shape (heigh, width, (R, G, B, A))
    """
    # puts these in the order should be concatenated (left to r)
    cam_names = [self.left_cam_name,
                 self.forward_cam_name,
                 self.right_cam_name,
                 self.backward_cam_name]
    sim_img_responses = self.request_all_4_sim_images(cam_names)
    return self.make_composite_image_from_responses(sim_img_responses)

  def get_car_state(self):
    """
    Get state of the car in the environment

    :returns: CarState object, which has as attributes: speed, handbrake,
    collision, kinematics_estimated 
    """
    
    return self.client.getCarState()

  def update_car_controls(self, car_controls):
    """
    Send new instructions to the car; update its controls.
    
    :param car_controls: airsim.CarControls() object

    :returns: nada
    """
    return client.setCarControls(car_controls)

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
      #img_2D_RGBA = np.flipud(img_1D.reshape(height,
      #                                       width,
      #                                       4))

      # But! CNN is spatial invariant, meaning it doesn't care if the
      # image is flipped or not, so no need to unflip it
      img_2D_RGBA = img_1D.reshape(height, width, 4)

      
      dict_of_2D_imgs.update({ cam_name : img_2D_RGBA})
      
    # customized to get a panoramic image w/ given camera orientations
    composite_img = np.concatenate([ dict_of_2D_imgs[cam_names[3]][:, int((2*width/5))::],
                                     dict_of_2D_imgs[cam_names[0]][:,int((2*width/5))::],
                                     dict_of_2D_imgs[cam_names[1]],
                                     dict_of_2D_imgs[cam_names[2]][:,0:int((3*width/5))],
                                     dict_of_2D_imgs[cam_names[3]][:,0:int((3*width/5))] ], axis=1)

    # for debugging and getting cameras right
    #airsim.write_png(os.path.normpath('sim_img'+ str(time.time())+'.png'), composite_img)

    return composite_img

  def emergency_reset(self):
    """
    Does not bother to pause the car before resetting;
    just tries to immediately reset. Good for when the car is
    falling through the map and need to catch it before it falls
    through.
    """
    print('emergency reset triggered')
    self.client.reset()

  def normal_reset(self):
    """
    Pauses the simulation before resetting the vehicle.
    """
    self.client.simPause(True)
    self.reset()
    self.simPause(False)
  

class DriverAgent():
  """
  The driver of the vehicle
  """
  
  def __init__(self, num_steering_angles, replay_memory_size,
               mini_batch_size=128, gamma=0.98):
    """
    
    """
    # state_t, action_t, reward_t, state_t+1
    self.replay_memory = [ (np.empty(INPUT_SHAPE, dtype=np.uint8),
                            0.0,
                            0.0,
                            np.empty(INPUT_SHAPE, dtype=np.uint8)) ] * replay_memory_size
                            
    self.action_space = np.arange(-1.0, 1.0000001, 2.0/num_steering_angles)
    self.num_steering_angles = num_steering_angles

    # this neural network will approximate the max_action(Q(state, action))
    # which approximates: max_action[ max(R) ]
    self.online_Q = Sequential()

    self.online_Q.add(Convolution2D(128, kernel_size=32, strides=28, input_shape=INPUT_SHAPE,
                                 data_format='channels_last', input_shape=INPUT_SHAPE))
    self.online_Q.add(Activation('relu'))
    self.online_Q.add(MaxPooling2D(pool_size=8, strides=8))

    self.online_Q.add(Flatten())

    self.online_Q.add(Dense(512))
    self.online_Q.add(Activation('relu'))
    self.online_Q.add(Dense(num_steering_angles))
    # output Q value for each possible action; pick action w/ highest Q value
    self.online_Q.add(Activation('linear'))
    
    self.online_Q.compile(optimizer='adam') # used for selecting actions -- copies weights from offline
    self.offline_Q = self.online_Q # target network -- used to update weights

    self.mini_batch_size = mini_batch_size
    self.gamma = gamma

  def reward_calc(self, collision_):
    pass

  def mini_batch_sample(self):
    pass

  def 
  def get_steering_angle(self, state):
    """
    Use the policy approximating neural network to calculate Q
    values for each of the possible steering angles. Pick the
    steering angle with the highest Q value.

    :param state: the composite image from airsim_env.get_composite_image()
    """


    # g

  def get_random_steering_angle(self):
    self.action_space[random.randint(0, self.num_steering_angles-1)]


    

# thank you: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html
def init_car_controls():
  car_controls = airsim.CarControls()
  car_controls.throttle = 0.3
  car_controls.steering = 0.0
  car_controls.is_manual_gear = True
  car_controls.manual_gear = 1  # should constrain speed

  return car_controls

def reward(car_state):
  """ :param car_state: airsim.CarState() of car in sim"""
  
  if car_state.collisions.has_collided is True:
    # collisions attrib has object_id (collided w/), so
    # very easily could modify and see if collided with person
    # or vehicle or building or curb or etc. 
    return -1.0
  else:
    return 1.0
  

print('Getting ready')
car_controls = init_car_controls()
replay_memory_size = 1500 # units=num images
episode_length = 2000
assert episode_length > replay_memory_size
num_episodes = 10
epsilon = 1.0  # probability of selecting a random action/steering angle
episodic_epsilon_linear_decay_amount = (epsilon / num_episodes) # decay to 0
num_steering_angles = 10
reward_delay = 0.05 # seconds until know assign reward

driver_agent = DriverAgent(num_steering_angles=num_steering_angles,
                           replay_memory_size=replay_memory_size)
airsim_env = AirSimEnv()


# for each episode (arbitrary length of time)
for episode_num in range(num_episodes):
  print('Starting episode {]'.format(episode_num+1))
  airsim_env.normal_reset()
  
  #  decay that epsilon value by const val
  if episode_num > 0:
    epsilon -= episodic_epsilon_linear_decay_amount

  # for each time step
  for time_step in range(episode_length):
    print('\t time step {}'.format(time_step+1))

    # Observation time_step+1
    car_state = airsim_env.get_car_state()

    # if car is starting to fall into oblivion
    if car_state.kinematics_estimated.position.z_val > -0.6:
      airsim_env.emergency_reset()
      
    composite_image = airsim_env.get_environment_state()
    driver_agent.replay_memory.remember_state(composite_image)

    # Actions
    # time to determine what to do
    # take random action w/ probability epislon (for exploration pusposes)
    if random.random() < epsilon:
      steering_angle = driver_agent.get_random_steering_angle(composite_image)
    else:
      # get an output from the NN and interpret it
      steering_angle = driver_agent.get_steering_angle(composite_image)

    car_controls.steering_angle = steering_angle
    airsim_env.update_car_controls(car_controls)
    
    driver_agent.replay_memory.remember_action(steering_angle)

    # REWARD
    time.sleep(reward_delay)

    # get the reward
    reward = airsim_env.get_car_state().collisions.has_collided
    
    driver_agent.replay_memory.remember_reward(reward)
    
    

    
  
  
clone net Q to obtain target Q' and use Q' for
generating the Q learning targets yj for the following C updates to Q

phi is just a preprocessing image func
