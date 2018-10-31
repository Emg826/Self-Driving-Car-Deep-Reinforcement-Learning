"""
Based on:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

import gym
from gym import spaces
import airsim
import numpy as np
import time
import random


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
    self.reward_delay = 0.05  # real-life seconds
    self.episode_step_count = 1
    self.steps_per_episode = (1/self.reward_delay) *  120 # right hand num is in in-game seconds 

    
    self.collisions_in_a_row = 0
    self.too_many_collisions_in_a_row = 15 # note: if stuck, then collisions will keep piling on
    self.obj_id_of_last_collision = -123456789
    
    self.client = airsim.CarClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)

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
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

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

    reward = self._get_reward(collision_info)

    # potential probelm of getting stuck, so track number of collisions in a row w/ same obj
    # measure in time steps rather than time because nn can take long, variable time to train
    # note: Landscape_1 is the side walk, which for some reason was given id=-1, same as
    # colliding w/ nothing
    if collision_info.object_id != -1 or 'Landscape_1' in collision_info.object_name:
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
      self.episode_step_count = 0
      self.collisions_in_a_row = 0
      self.obj_id_of_last_collision = -123456789
      done = True

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
    
    self.episode_start_time = time.time()
    self.client.simPause(False)
    
    return self._get_environment_state() # just to have an initial state?

  def render(self, mode='human'):
    pass  # airsim server binary handles rendering; we're just the client
    

  def _get_reward(self, collision_info):
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
    if collision_info.object_id != -1:  #-1 if not currently colliding
      return -1.0
    else:
      return 1.0
    
  def _get_environment_state(self):
    """
    Get state of the environment and the vehicle in the environment.

    :returns: panoramic image, numpy array with shape (heigh, width, (R, G, B, A))
    """
    # puts these in the order should be concatenated (left to r)
    cam_names = [self.left_cam_name,
                 self.forward_cam_name,
                 self.right_cam_name,
                 self.backward_cam_name]
    sim_img_responses = self._request_all_4_sim_images(cam_names)
    return self._make_composite_image_from_responses(sim_img_responses,
                                                     cam_names)

  def _request_all_4_sim_images(self, cam_names):
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
                                                           pixels_as_float=True,
                                                           compress=False) )

    return self.client.simGetImages(list_of_img_request_objs)

  def _make_composite_image_from_responses(self, sim_img_responses, cam_names):
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
    dict_of_2D_imgs = {}

    height = sim_img_responses[0].height
    width = sim_img_responses[0].width

    for cam_name, sim_img_response in zip(cam_names, sim_img_responses):
      dict_of_2D_imgs.update({cam_name : np.array(sim_img_response.image_data_float).reshape(height, width)})

    # customized to get a panoramic image w/ given camera orientations
    composite_img = np.concatenate([ dict_of_2D_imgs[cam_names[3]][:, int((2*width/5))::],
                                     dict_of_2D_imgs[cam_names[0]][:,int((2*width/5))::],
                                     dict_of_2D_imgs[cam_names[1]],
                                     dict_of_2D_imgs[cam_names[2]][:,0:int((3*width/5))],
                                     dict_of_2D_imgs[cam_names[3]][:,0:int((3*width/5))] ], axis=1)

    # for debugging and getting cameras right
    #airsim.write_png(os.path.normpath('sim_img'+ str(time.time())+'.png'), composite_img)

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

    
