"""
Based on:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

""" For settings.json

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

import gym
from gym import spaces
import airsim
import numpy as np
import time
import random
import os
import cv2
import math


class AirSimEnv(gym.Env):
  """Keras-rl usable gym (an openai package)"""

  def __init__(self):
    num_steering_angles = 5
    self.action_space = spaces.Discrete(num_steering_angles)
    self.action_space_steering = np.linspace(-1.0, 1.0, num_steering_angles).tolist()
    self.car_controls = airsim.CarControls(throttle=0.5125,
                                           steering=0.0,
                                           is_manual_gear=True,
                                           manual_gear=1)
    self.reward_delay = 0.2  # real-life seconds
    self.episode_step_count = 1.0  # 1.0 so that 
    self.steps_per_episode = (1/self.reward_delay) *  180 # right hand num is in in-game seconds 
    
    self.collisions_in_a_row = 0
    self.too_many_collisions_in_a_row = 15 # note: if stuck, then collisions will keep piling on
    self.obj_id_of_last_collision = -123456789  # anything <-1 is unassociated w/ an obj in sim (afaik)
    
    self.client = airsim.CarClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)

    self.distance_travelled = 0.0
    self.previous_x_coord = None  # init'd in reset(self)
    self.previous_y_coord = None

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'

    self._setup_my_cameras()
    # i just drove around and hit ';' and found the vectors and
    # quaternions of those locations
    # ; ordering: 1 2 3 4 while Quaternionr() has 2 3 4 1 for some reason
    # Note: these are all safe respawn points, i.e., not respawn into another vehicle
    self.reset_poses = [airsim.Pose(airsim.Vector3r(46,-530,-.7),
                                              airsim.Quaternionr(0,0,.01,1.02)), # checked
                                  airsim.Pose(airsim.Vector3r(320,-18,-.7),
                                              airsim.Quaternionr(0,0,1.0,.01)),  # checked
                                  airsim.Pose(airsim.Vector3r(229,-313,-.7),
                                              airsim.Quaternionr(0,0,-.73,.69)),   # checked
                                  airsim.Pose(airsim.Vector3r(-800,-4,-.7),
                                              airsim.Quaternionr(0,0,.02,1.0)),   # checked
                                   airsim.Pose(airsim.Vector3r(-138.292, -7.577, -0.7),
                                                   airsim.Quaternionr(0.0, 0.0, 0.002, 1.0)),
                                   airsim.Pose(airsim.Vector3r(68.873, 160.916, -1.05),
                                                   airsim.Quaternionr(0.0, 0.0, -.7, 0.714)),
                                   airsim.Pose(airsim.Vector3r(55.514, -310.598, -1.05),
                                                   airsim.Quaternionr(0.0, 0.0, .707, .707)),
                                   airsim.Pose(airsim.Vector3r(64.665, -352.195, -1.05),
                                                   airsim.Quaternionr(0.0, 0.0, .717, .697)),
                                   airsim.Pose(airsim.Vector3r(219.288, 201.129, -1.05),
                                                   airsim.Quaternionr(0.0, 0.0, -.383, .924)),
                                   airsim.Pose(airsim.Vector3r(67.507, 234.912, -1.05),
                                                   airsim.Quaternionr(0.0, 0.0, -.7, 0.715))]


  def step(self, action):
    """
    The main logic for interacting with the simulation. This is where actions
    are submitted, states and rewards are acquired, and error checking (falling into
    oblivion, spirialing out of control, getting stuck, etc.) are all done.

    :param action: an idx (so an integer) that corresponds to an action in the action_space.
    this idx comes from the policy, i.e., from random selection or from ddqn.

    :returns: standard? openai gym stuff for step() function
    """
    #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    # action_t
    steering_angle = self.action_space_steering[action]
    self.car_controls.steering = steering_angle
    
    
    self.client.simPause(False)  # unpause AirSim

    self.client.setCarControls(self.car_controls)
    time.sleep(self.reward_delay)

    # reward_t
    collision_info = self.client.simGetCollisionInfo()
    car_info = self.client.getCarState()

    state_t2 = self._get_environment_state()

    self.client.simPause(True)  # pause to do backend stuff

    current_x = car_info.kinematics_estimated.position.x_val
    current_y = car_info.kinematics_estimated.position.y_val
    # z is down+ and up-, so not need

    # euclidean distance since last step
    self.distance_travelled += math.sqrt( (current_x - self.previous_x_coord)**2 \
                                         +(current_y - self.previous_y_coord)**2 )

    reward = self._get_reward(collision_info, car_info)

    # potential probelm of getting stuck, so track number of collisions in a row w/ same obj
    # measure in time steps rather than time because nn can take long, variable time to train
    # note: Landscape_1 is the side walk, which for some reason was given id=-1, same as
    # colliding w/ nothing
    #print(collision_info.has_collided, collision_info.object_id)  # debug
    #print(collision_info.object_name)   # debug
    if collision_info.object_name is not '' and car_info.speed < 1.0:
      if self.obj_id_of_last_collision == collision_info.object_id:
        self.collisions_in_a_row += 1
      else:
        self.collisions_in_a_row = 1
      self.obj_id_of_last_collision = collision_info.object_id
        
    # done if episode timer runs out (1) OR if fallen into oblilvion (2)
    # OR if spinning out of control (3) OR if knocked into the stratosphere (4)
    done = False

    if self.episode_step_count >  self.steps_per_episode or \
       car_info.kinematics_estimated.position.z_val > -0.5125 or \
       abs(car_info.kinematics_estimated.orientation.y_val) > 0.3125 or \
       car_info.speed > 40.0 or \
       self.collisions_in_a_row > self.too_many_collisions_in_a_row:
      self.episode_step_count = 1.0
      self.collisions_in_a_row = 0
      self.obj_id_of_last_collision = -123456789
      done = True

    self.previous_x_coord = current_x
    self.previous_y_coord = current_y
    
    return state_t2, reward, done, {}

  def reset(self):
    """
    When done (in step()) is true, this function is called.

    :returns: what the cartpole example returns (standard? for openai gym env.reset())
    """
    self.client.simPause(True)
    self.client.armDisarm(True)
    self.client.reset()
    reset_pose = self.reset_poses[random.randint(0, len(self.reset_poses)-1)]
    
    self.client.simSetVehiclePose(pose=reset_pose,
                                  ignore_collison=True)
    self.distance_travelled = 0.0
    self.previous_x_coord = reset_pose.position.x_val
    self.previous_y_coord = reset_pose.position.y_val
    
    self.episode_start_time = time.time()
    self.client.simPause(False)
    
    return self._get_environment_state() # just to have an initial state?

  def render(self, mode='human'):
    pass  # airsim server binary handles rendering; we're just the client
    

  def _get_reward(self, collision_info, car_info):
    """
    Calculate the reward for this current time step based on whether or not
    the car is colliding with something. This reward function does not use
    collision_info.has_collided because that attribute tells only if collision since car last
    reset, i.e., after stop colliding, has_collided remains true. Does not discriminate
    between people or cars or buildings (could do so w/ object_name attribute). Clips
    reward [-1, 1] just like in DQN paper.

    :param collision_info: an airsim.CollisionInfo() object
    :returns: floating point number, [-1, 1], of the reward for current time step

    """
    # collision id is finnicky; sometimes it will not register
    # like has_collided, object_name does not reset to '' after uncollide, but it does tell if colliding w/ new obj
    # and when object_id does not register, so it is more useful than has_collided
    if collision_info.object_id != -1 or \
       (collision_info.object_id == -1 and collision_info.object_name != '' and car_info.speed < 1.0):  #-1 if not currently colliding
      return -1.0
    else:
      # w_dist * (sigmoid(sqrt( 0.15*x)- w_dist*10)
      w_dist = 0.65
      exponent = -1*(math.sqrt(0.15*self.distance_travelled) - (10*w_dist))  # @ 0.15*dist_trav: hit 0.6 reward @ 500units
      total_distance_contrib = w_dist * (1 / (1 + math.exp(exponent)))

      # slight reward for steering straight, i.e., only turn if necessary in long term
      w_non0_steering = 0.1
      steering_contrib = w_non0_steering * math.cos(self.car_controls.steering)

      w_steps = 1.0 - w_non0_steering - w_dist
      step_contrib = w_steps * (self.episode_step_count / self.steps_per_episode)
      

      # NOTE: w_dist + w_non0_steering + w_steps  <= 1
                                   
      return total_distance_contrib + steering_contrib + step_contrib

  
    
  def _get_environment_state(self):
    """
    Get state of the environment and the vehicle in the environment.

    :returns: panoramic image, numpy array with shape (heigh, width, (R, G, B, A))
    """
    # puts these in the order should be concatenated (left to r)
    cam_names = [self.left_cam_name,
                 self.forward_cam_name,
                 self.right_cam_name]
    sim_img_responses = self._request_all_sim_images(cam_names)
    return self._make_composite_image_from_responses(sim_img_responses)

  def _request_all_sim_images(self, cam_names):
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

  def _make_composite_image_from_responses(self, sim_img_responses):
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

    height = sim_img_responses[0].height
    width = sim_img_responses[0].width
    # byte string array --> 1d numpy array --> png --> add to composite image
    preprocess_individual = lambda img_response_obj:  cv2.cvtColor(np.fromstring(img_response_obj.image_data_uint8, dtype=np.uint8).reshape(height, width, 4),
                                                                                          cv2.COLOR_BGR2GRAY) * 1.2 # 1.2x  brightness
    preprocessed_individual_imgs = []
    for img_response_obj in sim_img_responses:
      preprocessed_individual_imgs.append(preprocess_individual(img_response_obj)*2.0)

    composite_img = np.concatenate([preprocessed_individual_imgs[0][int(3*height/7)::,int((2*width/5))::],
                                                  preprocessed_individual_imgs[1][int(3*height/7)::,:],
                                                  preprocessed_individual_imgs[2][int(3*height/7)::,0:int((3*width/5))]], axis=1)

    # for debugging and getting cameras right
    #cv2.imwrite('{}.jpg'.format(time.time()), composite_img)

    return composite_img

  def _setup_my_cameras(self):
    """
    Helper function to set the left, right, forward, and back cameras up
    on the vehicle as I've see fit.

    :returns: nada
    """
    # creates a panoram-ish camera set-up
    self.client.simSetCameraOrientation(self.left_cam_name,
                                        airsim.Vector3r(0.0, 0.0, -0.68))
    self.client.simSetCameraOrientation(self.right_cam_name,
                                   airsim.Vector3r(0.0, 0.0, 0.68))
    self.client.simSetCameraOrientation(self.forward_cam_name,
                                        airsim.Vector3r(0.0, 0.0, 0.0))
    self.client.simSetCameraOrientation(self.backward_cam_name,
                                        airsim.Vector3r(0.0, 0.0, 11.5))

    
