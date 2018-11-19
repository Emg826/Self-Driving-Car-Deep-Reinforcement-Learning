"""
Based on:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

""" Copy and paste this into your settings.json (which hould be in your Documents folder)

{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "RpcEnabled": true,
  "EngineSound": false,
  "ClockSpeed": 1,
  "CameraDefaults": {
    "CaptureSettings": [{
      "ImageType": 0,
      "Width": 640,
      "Height": 384,
      "FOV_Degrees": 120
    },
    {
      "ImageType": 1,
      "Width": 640
      "Height": 384
      "FOV_Degrees": 120
    }],
    "X": -10, "Y": 0, "Z": -0.5,
    "Pitch": -3, "Roll": 0, "Yaw": 0
  }           
}

"""

from gym import spaces, Env
import airsim
import numpy as np
import random
import os
import cv2
import math
import queue
import time # just to check steps per second 



class AirSimEnv(Env):
  """Keras-rl usable gym (an openai package)"""

  def __init__(self, num_steering_angles, max_num_steps_in_episode,
                   time_steps_between_dist_calc,
                   settings_json_image_w,
                   settings_json_image_h,
                   fraction_of_top_of_img_to_cutoff,
                   fraction_of_bottom_of_img_to_cutoff,
                   lambda_function_to_apply_to_pixels= lambda pixel_value: pixel_value):
    """
    Note: preprocessing_lambda_function_to_apply_to_pixels is applied to each pixel,
    and the looping through the image is handled by this class. Therefore, only 1
    parameter to this lambda function, call it pixel_value or something. Default is
    a do nothing function.
    """
    # image stuff
    self.PHI = lambda_function_to_apply_to_pixels  # PHI from DQN algorithm

    self.first_row_idx = int(settings_json_image_h * fraction_of_top_of_img_to_cutoff)
    self.last_row_idx = int(settings_json_image_h * (1-fraction_of_bottom_of_img_to_cutoff))
    
    assert self.first_row_idx < self.last_row_idx
    
    self.img_shape = (self.last_row_idx-self.first_row_idx, settings_json_image_w)
   
    # steering stuff
    self.action_space = spaces.Discrete(num_steering_angles)
    self.action_space_steering = np.linspace(-1.0, 1.0, num_steering_angles).tolist()
    self.car_controls = airsim.CarControls(throttle=0.50,
                                                       steering=0.0,
                                                       is_manual_gear=True,
                                                       manual_gear=1)

    # in-sim episode handling
    self.episode_step_count = 0
    self.steps_per_episode = max_num_steps_in_episode # right hand num is in in-game seconds

    # collision info for emergency resets and reward func calc
    self.collisions_in_a_row = 0
    self.too_many_collisions_in_a_row = 3 # note: if stuck, then collisions will keep piling on
    # also collisions being stuck can cause glitch through map 
    
    self.obj_id_of_last_collision = -123456789  # anything <-1 is unassociated w/ an obj in sim (afaik)

    # connect to the client; we're ready to connect to set up our cameras
    self.client = airsim.CarClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)

    # major problem: car drives around in circle; need space out distance travelled
    self.distance_travelled = 0.0
    self.coords_offset = time_steps_between_dist_calc  # num steps ago to calculate distance travelled from
    self.coords_queue = queue.Queue(self.coords_offset)  # stores (x, y) coordinate tuples

    self.time_since_ep_start = 2**31  # only used for debug and steps/second measurement,
    # not for tracking progress in sim; note: sim steps per real life seconds is ~6

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'

    self._setup_my_cameras()
    # i just drove around and hit ';' and found the vectors and
    # quaternions of those locations
    # ; ordering: 1 2 3 4 while Quaternionr() has 2 3 4 1 for some reason
    # Note: these are all safe respawn points, i.e., not respawn into another vehicle
    self.reset_poses = [airsim.Pose(airsim.Vector3r(-727.012,-7.407,-.7),  # safe
                                                  airsim.Quaternionr(0,0,.016,1.0)),
                              airsim.Pose(airsim.Vector3r(-169.546, -316.207, -0.72), # safe
                                              airsim.Quaternionr(0.0, 0.0, .712, 0.702)),
                              airsim.Pose(airsim.Vector3r(-292.415, 32.229, -0.7), # safe
                                              airsim.Quaternionr(0.0, 0.0, -.679, 0.734)),
                              airsim.Pose(airsim.Vector3r(222.984, -491.947, -0.7), # safe
                                              airsim.Quaternionr(0.0, 0.0, .716, .698)),
                              airsim.Pose(airsim.Vector3r(311.298, -10.177, -0.688), # safe
                                              airsim.Quaternionr(0.0, 0.0, -1.0,.006)),
                              airsim.Pose(airsim.Vector3r(-191.452, -474.923, -0.689), # safe
                                              airsim.Quaternionr(0.0, 0.0, .008,1.0)),
                              airsim.Pose(airsim.Vector3r(-316.513, 144.906, -0.688), # safe
                                              airsim.Quaternionr(0.0, 0.0, -1.0,.003))]


  def step(self, action):
    """
    The main logic for interacting with the simulation. This is where actions
    are submitted, states and rewards are acquired, and error checking (falling into
    oblivion, spirialing out of control, getting stuck, etc.) are all done.

    :param action: an idx (so an integer) that corresponds to an action in the action_space.
    this idx comes from the policy, i.e., from rssssandom selection or from ddqn.

    :returns: standard? openai gym stuff for step() function
    """
    #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    # action_t
    steering_angle = self.action_space_steering[action]
    self.car_controls.steering = steering_angle

    self.client.simPause(False)  # unpause AirSim

    self.client.setCarControls(self.car_controls)

    # reward_t
    state_t2 = self._get_environment_state()
    collision_info = self.client.simGetCollisionInfo()
    car_info = self.client.getCarState()

    self.client.simPause(True)  # pause to do backend stuff

    current_x = car_info.kinematics_estimated.position.x_val
    current_y = car_info.kinematics_estimated.position.y_val
    # z is down+ and up-, so not need

    # euclidean distance since last n step
    if self.coords_queue.full(): # only get when full bcuz want larger distance over time
      past_x, past_y = self.coords_queue.get()
      self.distance_travelled += math.sqrt( (current_x - past_x)**2 \
                                                       +(current_y - past_y)**2 )
    self.coords_queue.put( (current_x, current_y) )

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
       car_info.kinematics_estimated.position.z_val > -0.55 or \
       abs(car_info.kinematics_estimated.orientation.y_val) > 0.3125 or \
       car_info.speed > 25.0 or \
       self.collisions_in_a_row > self.too_many_collisions_in_a_row or \
       car_info.kinematics_estimated.position.z_val < -2.5:
      done = True

    self.episode_step_count += 1


    # for debug
    #if self.episode_step_count % 30 == 0:
    #  print('Ep step {}, averaging {} steps per IRL sec'.format(self.episode_step_count,
    #                                                                              (self.episode_step_count / (time.time() -self.time_since_ep_start))))
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
    print(reset_pose)

    while not self.coords_queue.empty():  # clear the queue
      _ = self.coords_queue.get()

    self.episode_step_count = 0
    self.collisions_in_a_row = 0
    self.obj_id_of_last_collision = -123456789 # any int < -1 is ok

    self.client.simPause(False)
    self.time_since_ep_start = time.time()  # again, for debug purposes 

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
      w_dist = 0.965
      assert w_dist <= 1.0

      # hit 1.0 reward @ 500units
      total_distance_contrib = w_dist * (self.distance_travelled / 500 )

      # slight reward for steering straight, i.e., only turn if necessary in long term
      w_non0_steering = 1.0-w_dist

      # note: steering [-1, 1], so w/ 5 steering angles, have: {-1., -0.5, 0., .5, 1.}
      # this means the decrease in reward (never penalty though) is linear deviate from 0. steering
      steering_contrib = w_non0_steering * (1 - abs(self.car_controls.steering))

      return total_distance_contrib + steering_contrib

  def _get_environment_state(self):
    """
    Get state of the environment and the vehicle in the environment.

    :returns: panoramic image, numpy array with shape (heigh, width, (R, G, B, A))
    """
    # puts these in the order should be concatenated (left to r)
                
    sim_img_responses = self._request_sim_images([self.forward_cam_name])
    return self._make_preprocessed_depth_planner_image(sim_img_responses)

  def _request_sim_images(self, cam_names):
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
      list_of_img_request_objs.append(airsim.ImageRequest(camera_name=cam_name,
                                                           image_type=airsim.ImageType.DepthPlanner,
                                                           pixels_as_float=True,
                                                           compress=False))

    return self.client.simGetImages(list_of_img_request_objs)

  def _make_preprocessed_depth_planner_image(self, sim_img_response_list):
    """
    Apply self.PHI to a depth planner image. Expects only 1 image response
    in sim_img_response_list, and that type should be type airsim.DepthPlanner """
    height = sim_img_response_list[0].height
    width = sim_img_response_list[0].width

    # originally, the image is a 1D python list; want to turn it into a 2D numpy array
    img = airsim.list_to_2d_float_array(sim_img_response_list[0].image_data_float,
                                                   width, height)
    # only need middle 60% of the image
    img = img[self.first_row_idx : self.last_row_idx]

    # apply PHI to each pixel - can only do if 2 dimension, i.e. grayscale
    if len(img.shape) ==  2:
      for row_idx in range(0, img.shape[0]):
        for col_idx in  range(0, img.shape[1]):
          img[row_idx][col_idx] = self.PHI(img[row_idx][col_idx])

    else:
      print('Warning! image could not be preprocessed')

    # for debugging and getting cameras correct
    #cv2.imwrite('{}.jpg'.format(time.time()), img)
    
    return img

  def _make_composite_image_from_responses(self, sim_img_responses):
    """
    Take the list of ImageResponse objects (the observation), 2D numpy array,
    and then concatenate all of the images into a composite (panoramic-ish) image.

    Note: expects 3 img response obj, a left, front, and right img response objs,
    in that order.

    :param sim_img_responses: list of ImageResponse obj from a call to
    request_all_4_sim_images() in the AirSimEnv class; Note: should be type DepthPlanner
    
    :returns: 2D channel numpy array of all images "stitched" together and after
    applying self.PHI (pixel by pixel preprocessing function) 
    """
    # 4 channel images; append that image to a list

    height = sim_img_responses[0].height
    width = sim_img_responses[0].width

    # originally, the image is a 1D python list; want to turn it into a 2D numpy array
    reshaped_imgs = []
    for img_response_obj in sim_img_responses:
      reshaped_imgs.append(airsim.list_to_2d_float_array(img_response_obj.image_data_float,
                                                                            img_response_obj.width,
                                                                            img_response_obj.height))

    # stitch left, front, and right camera images together
    composite_img = np.concatenate([reshaped_imgs[0][int(3*height/7)::,int((2*width/5))::],
                                                  reshaped_imgs[1][int(3*height/7)::,:],
                                                  reshaped_imgs[2][int(3*height/7)::,0:int((3*width/5))]], axis=1)

    # apply PHI to each pixel - can only do if 2 dimension, i.e. grayscale
    if len(composite_img.shape) ==  2:
      for row_idx in range(0, composite_img.shape[0]):
        for col_idx in  range(0, composite_img.shape[1]):
          composite_img[row_idx][col_idx] = self.PHI(composite_img[row_idx][col_idx])

    # else, maybe there's an RGB value in each pixel location?
    elif len(composite_img.shape) == 3:
      for row_idx in range(0, composite_img.shape[0]):
        for col_idx in  range(0, composite_img.shape[1]):
          for _ in range(0, composite_img.shape[2]):
            composite_img[row_idx][col_idx][_] = self.PHI(composite_img[row_idx][col_idx][_])
    else:
      print('Warning! image could not be preprocessed')

    # for debugging and getting cameras correct
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
