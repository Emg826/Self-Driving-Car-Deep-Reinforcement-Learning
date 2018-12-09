"""
Based on:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

""" Copy and paste this into your settings.json (which hould be in your Documents folder)
{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "RpcEnabled": true,
  "ViewMode": "SpringArmChase",
  "EngineSound": false,
  "ClockSpeed": 1.0,
  "CameraDefaults": {
    "CaptureSettings": [{
      "ImageType": 0,
      "Width": 1024,
      "Height": 512,
      "FOV_Degrees": 120
    },
    {
      "ImageType": 1,
      "Width": 512,
      "Height": 256,
      "FOV_Degrees": 120
    }]
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


SCENE_INPUT_SHAPE = (512, 1024)
DEPTH_PLANNER_INPUT_SHAPE = (256, 512)
SENSOR_INPUT_SHAPE = (17,)

class AirSimEnv(Env):
  """Keras-rl usable gym (an openai package)"""

  def __init__(self,
                  num_steering_angles,
                  max_num_steps_in_episode=10**4,
                  fraction_of_top_of_scene_to_drop=0.0,
                  fraction_of_bottom_of_scene_to_drop=0.0,
                  fraction_of_top_of_depth_to_drop=0.0,
                  fraction_of_bottom_of_depth_to_drop=0.0,
                  seconds_pause_between_steps=0.03,  # gives rand num generator time to work (wasn't working b4)
                  seconds_between_collision_in_sim_and_register=1.5,  # note avg 4.12 steps per IRL sec on school computer
                  lambda_function_to_apply_to_depth_pixels=None):
    """
    Note: preprocessing_lambda_function_to_apply_to_pixels is applied to each pixel,
    and the looping through the image is handled by this class. Therefore, only 1
    parameter to this lambda function, call it pixel_value or something. Default is
    a do nothing function.
    """
    # sim admin stuff
    self.seconds_pause_between_steps = seconds_pause_between_steps
    self.seconds_between_collision_in_sim_and_register = seconds_between_collision_in_sim_and_register
    
    # image stuff
    self.PHI = lambda_function_to_apply_to_depth_pixels  # PHI from DQN algorithm


    self.first_scene_row_idx = int(SCENE_INPUT_SHAPE[0] * fraction_of_top_of_scene_to_drop)
    self.last_scene_row_idx = int(SCENE_INPUT_SHAPE[0] * (1-fraction_of_bottom_of_scene_to_drop))

    self.first_depth_planner_row_idx = int(DEPTH_PLANNER_INPUT_SHAPE[0] * fraction_of_top_of_depth_to_drop)
    self.last_depth_planner_row_idx = int(DEPTH_PLANNER_INPUT_SHAPE[0] * (1-fraction_of_bottom_of_depth_to_drop))

    
    
    assert self.first_scene_row_idx < self.last_scene_row_idx
    assert self.first_depth_planner_row_idx < self.last_depth_planner_row_idx


    # note the shapes of inputs to the neural network; can retrieve in keras_drl_steering... .py
    self.scene_input_shape = (self.last_scene_row_idx-self.first_scene_row_idx, SCENE_INPUT_SHAPE[1])
    self.depth_planner_input_shape = (self.last_depth_planner_row_idx-self.first_depth_planner_row_idx, DEPTH_PLANNER_INPUT_SHAPE[1])
    self.sensor_input_shape = SENSOR_INPUT_SHAPE

    
    # steering stuff
    self.action_space = spaces.Discrete(num_steering_angles)
    self.action_space_steering = np.linspace(-1.0, 1.0, num_steering_angles).tolist()
    self.car_controls = airsim.CarControls(throttle=0.50,
                                                       steering=0.0,
                                                       is_manual_gear=True,
                                                       manual_gear=1)  # should constrain speed to < 18ish mph
    # in-sim episode handling
    self.episode_step_count = 0
    self.steps_per_episode = max_num_steps_in_episode 

    # collision info for emergency resets and reward func calc
    self.collisions_in_a_row = 0
    self.max_acceptable_collisions_in_a_row = 3 # note: if stuck, then collisions will keep piling on
    # also collisions being stuck can cause glitch through map 
    
    self.obj_id_of_last_collision = -123456789  # anything <-1 is unassociated w/ an obj in sim (afaik)

    # connect to the client; we're ready to connect to set up our cameras
    self.client = airsim.CarClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    
    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'

    self._setup_my_cameras()

    self.scene_request_obj = airsim.ImageRequest(camera_name='0',
                                                                   image_type=airsim.ImageType.Scene,
                                                                   pixels_as_float=False,
                                                                   compress=False)
    
    self.depth_planner_request_obj = airsim.ImageRequest(camera_name='0',
                                                                             image_type=airsim.ImageType.DepthPlanner,
                                                                             pixels_as_float=True,
                                                                             compress=False)
    self.list_of_img_request_objects = [self.scene_request_obj, self.depth_planner_request_obj]


    # ; ordering: 1 2 3 4 while Quaternionr() has 2 3 4 1 for some reason
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
                                              airsim.Quaternionr(0.0, 0.0, -1.0, 0.003)),
                              airsim.Pose(airsim.Vector3r(16.631, -38.537, -.9), # safe - aim @ roundabout center and can go L or R
                                              airsim.Quaternionr(-0.005, 0.011, 0.296, 0.955)),
                              airsim.Pose(airsim.Vector3r(-316.513, 144.906, -0.688), # safe - aim @ roundabout but better to go L
                                              airsim.Quaternionr(0.0, 0.0, -1.0,.003)),
                              airsim.Pose(airsim.Vector3r(33.322, -528.89, -0.688), # safe
                                              airsim.Quaternionr(0.0, 0.0, -0.002,1.0))]

    #  instead of driving about aimlessly, will try to arrive to 1 destination from 1 starting point
    self.beginning_coords = airsim.Pose(airsim.Vector3r(316.574,-2.870,-.7),  # safe
                                                  airsim.Quaternionr(0.0,0.0,-0.967,0.253))
    self.ending_coords = airsim.Vector3r(73.728,-51.480,-.7)  # appx 298 m from start; mostly straight


    # units are meters
    self.ending_circle_radius = 8.0 # if car in circle w/ this radius, then car has arrived @ destination

    # stuff used in reward function
    self.total_distance_to_destination = self._manhattan_distance(self.ending_coords.x_val,
                                                                                         self.beginning_coords.position.x_val ,
                                                                                         self.ending_coords.y_val,
                                                                                         self.beginning_coords.position.y_val)
    self.current_distance_from_destination = self.total_distance_to_destination
    self.episode_time_in_simulation_secs = 1.0   # tracks time between unpause and pause in step; used
    self.current_distance_travelled_towards_destination = 0.0
    self.episode_time_in_irl_seconds = 0.0  # only used for debug and steps/second measurement,
    # not for tracking progress in sim; note: sim steps per real life seconds is ~6

    print('The car will need to travel {} simulation meters to arrive at the destination'.format(self.total_distance_to_destination))


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

    # BEGIN TIMESTEP
    self.client.simPause(False)  # unpause AirSim
    sim_unpaused_start_time = time.time()  # used to track time at which sim is unpaused

    self.client.setCarControls(self.car_controls)

    time.sleep(self.seconds_pause_between_steps)  # take this action for user specified amt of IRL seconds

    # can't get this information once pause, so get it now
    list_of_img_response_objects = self._request_sim_images()
    collision_info = self.client.simGetCollisionInfo()
    car_info = self.client.getCarState()

    self.client.simPause(True)  # pause to do backend stuff
    self.episode_time_in_simulation_secs += time.time() - sim_unpaused_start_time

    # END TIMESTEP
    # BEGIN BACKEND   # reward_t and state_t2

    # used in reward and state making
    self.current_distance_from_destination =  self._manhattan_distance(car_info.kinematics_estimated.position.x_val,
                                                                                                self.ending_coords.x_val,
                                                                                                 car_info.kinematics_estimated.position.y_val,
                                                                                                 self.ending_coords.y_val)
    self.current_distance_travelled_from_origin = self._manhattan_distance(car_info.kinematics_estimated.position.x_val,
                                                                                                self.beginning_coords.position.x_val ,
                                                                                                 car_info.kinematics_estimated.position.y_val,
                                                                                                 self.beginning_coords.position.y_val )
    self.current_distance_travelled_towards_destination = self.total_distance_to_destination - self.current_distance_from_destination

    reward_t = self._get_reward(collision_info, car_info)

    statet2 = self._make_state(car_info, list_of_img_response_objects)

    # track collisions in a row for checking if stuck [e.g., on a curb]
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
       car_info.speed > 17.0 or \
       self.collisions_in_a_row > self.max_acceptable_collisions_in_a_row or \
       car_info.kinematics_estimated.position.z_val < -2.5:
      done = True
      reward = -1.0

    self.episode_step_count += 1

    # mostly for debug, but can be helpful
    if self.episode_step_count % 30 == 0:
      print('Ep step {}, averaging {} steps per IRL sec'.format(self.episode_step_count,
                                                                                  (self.episode_step_count / (time.time() -self.episode_time_in_irl_seconds))))
      
    # check if made it to w/in the radius of the destination area/circle
    if self._arrived_at_destination(car_info):
      done = True
      reward = 1.0
      
    return state_t2, reward, done, {}
  

  def reset(self):
    """
    When done (in step()) is true, this function is called.

    :returns: what the cartpole example returns (standard? for openai gym env.reset())
    """
    self.client.simPause(True)
    self.client.armDisarm(True)
    self.client.reset()
    #reset_pose = self.reset_poses[random.randint(0, len(self.reset_poses)-1)]  # from when would drive aimlessly

    self.client.simSetVehiclePose(pose=self.beginning_coords,
                                           ignore_collison=True)

    self.episode_step_count = 0
    self.collisions_in_a_row = 0
    self.obj_id_of_last_collision = -123456789 # any int < -1 is ok

    self.episode_time_in_simulation_secs = 0.1  # avoid div by 0 err when call _make_state @ fin 

    self.client.simPause(False)
    self.episode_time_in_irl_seconds = time.time()  # again, for debug purposes
    self.current_distance_from_destination = self.total_distance_to_destination
    self.episode_time_in_simulation_secs = 1.0
  
    list_of_img_response_objects = self._request_sim_images()
    car_info = self.client.getCarState()

    # initial state for this new episode
    return self._make_state(car_info, list_of_img_response_objects)  


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
    reward = 0.0

    # id -1 if unnamed obj; not imply not colliding, just unnamed
    # if collided with something with a name
    sec_since_last_collision = time.time() -  collision_info.time_stamp*10**(-9) 
    if sec_since_last_collision < self.seconds_between_collision_in_sim_and_register and collision_info.time_stamp != 0.0:  # irl seconds
      self.distance_since_last_collision = 0.0
      reward = -1.0

    # if hit curb or an unnamed object
    elif (collision_info.object_id == -1 and collision_info.object_name != '' and car_info.speed < 1.0) or \
         (abs(car_info.kinematics_estimated.orientation.x_val) > 0.035 or abs(car_info.kinematics_estimated.orientation.y_val) > 0.035):   # check if hit curb (car x and y orientation changes)
      self.distance_since_last_collision = 0.0
      reward = -0.1
     
    # if have made very little progress towards goal so far -- note, get about 3 to 4 steps per IRL sec on school computer
    elif (self.total_distance_to_destination - self.current_distance_from_destination) <  50.0 and self.episode_step_count > 100:
      reward = -1.0

    else:
      """
      # From when car would drive aimlessly

      # w_dist * (sigmoid(sqrt( 0.15*x)- w_dist*10)
      w_dist = 0.970
      assert w_dist <= 1.0

      # hit 1.0 reward @ 2kunits
      total_distance_contrib = w_dist * max(0, min(1.0, (self.distance_since_last_collision / 300 )))

      # slight reward for steering straight, i.e., only turn if necessary in long term
      w_non0_steering = 1.0-w_dist

      # note: steering [-1, 1], so w/ 5 steering angles, have: {-1., -0.5, 0., .5, 1.}
      # this means the decrease in reward (never penalty though) is linear deviate from 0. steering
      steering_contrib = w_non0_steering * (1.0 - abs(self.car_controls.steering))


      # for debug
      #print(self.distance_since_last_collision, total_distance_contrib + steering_contrib)
      return total_distance_contrib + steering_contrib
      """
      current_average_meters_per_sim_secs = self.current_distance_travelled_from_origin /  self.episode_time_in_simulation_secs

      # time = distance / rate --- time to get from origin to end, at the average pace of the vehicle
      current_estimate_of_sim_secs_from_beginning_get_to_destination = self.total_distance_to_destination / current_average_meters_per_sim_secs

      # will go negative 
      sim_secs_remaining_to_get_to_destination = current_estimate_of_sim_secs_from_beginning_get_to_destination - self.episode_time_in_simulation_secs

      # will always be <= 0
      time_reward = max(-1.0, min(0, sim_secs_remaining_to_get_to_destination / current_estimate_of_sim_secs_from_beginning_get_to_destination))    # 1 - proportion saying how far along the car is to arriving @ destination

      # will always be >= 0
      distance_reward = max(0.0, self.current_distance_travelled_towards_destination**4 / self.total_distance_to_destination**4)

      reward = time_reward + distance_reward
      
    # for debug
    print('reward', reward)
    return reward


  def _make_state(self, car_info, list_of_img_response_objects):
    """
    :param car_info: returned by airsim.getCarState(), so is airsim.CarState object
    :param list_of_img_response_objects: list of length 2; @idx=0 is scene airsim.ImageResponse, and @idx=1
        is airsim.ImageResponse for depth planner
    :returns: statet2 (a list in ordered as model from keras_drl_... .py expects), reward_t (which is a float [-1, 1])

    Note: assumes that  self.current_distance_from_destination has been updated already
    """

    state_t2 = []
    state_t2.append(self._extract_scene_image(list_of_img_response_objects))
    state_t2.append(self._extract_depth_planner_image(list_of_img_response_objects))
    state_t2.append(self._extract_sensor_data(car_info))
    print(len(state_t2))
    return [1, 2, 3]  # order should be: scene img, depth img, sensor data


  
  def _extract_sensor_data(self, car_info):
    """
    Returns a list w/ 17 entries: 

    1-2. GPS (x, y) coordinates of car
    3. manhattan distance from end point (x, y)
    4. yaw/compass direction of car in radians  # use airsim.to_eularian_angles() to get yaw
    5. relative bearing in radians (see func)
    6. current steering angle in [-1.0, 1.0]
    x. linear velocity (x, y) # no accurate whatsoever (press ';' in sim to see)
    7-8. angular velocity (x, y)
    9-10. linear acceleration (x, y)
    11-12. angular acceleration (x, y)
    13. speed
    14-15. absolute difference from current location to destination for x and y each
    16-17. (x, y) coordinates of destination

    Note: no z coords because city is almost entirely flat
    """
    sensor_data = []
    
    # 1-2 car's gps x and y coords
    sensor_data.append(car_info.kinematics_estimated.position.x_val)
    sensor_data.append(car_info.kinematics_estimated.position.y_val)

    # 3 manhattan distance til end
    sensor_data.append(self.current_distance_from_destination)

    # 4 yaw, which is relative to the world frame, so can translatie invariant?
    yaw = airsim.to_eularian_angles(car_info.kinematics_estimated.orientation)[2]  # (pitch, roll, yaw)
    sensor_data.append(yaw)
      
     # 5 relative bearing
    sensor_data.append(self._relative_bearing(yaw,
                                                            (car_info.kinematics_estimated.position.x_val, car_info.kinematics_estimated.position.y_val),
                                                            (self.ending_coords.x_val, self.ending_coords.y_val)) )

    # 6 steering angle
    sensor_data.append(self.car_controls.steering)

    # 7 angular velocity x and y
    sensor_data.append(car_info.kinematics_estimated.angular_velocity.x_val)
    sensor_data.append(car_info.kinematics_estimated.angular_velocity.y_val)

    # 9 
    sensor_data.append(car_info.kinematics_estimated.linear_acceleration.x_val)
    sensor_data.append(car_info.kinematics_estimated.linear_acceleration.y_val)

    # 11
    sensor_data.append(car_info.kinematics_estimated.angular_acceleration.x_val)
    sensor_data.append(car_info.kinematics_estimated.angular_acceleration.y_val)

    # 13
    sensor_data.append(car_info.speed)

    # 14
    sensor_data.append(abs(self.ending_coords.x_val - car_info.kinematics_estimated.position.x_val))
    sensor_data.append(abs(self.ending_coords.y_val - car_info.kinematics_estimated.position.y_val))

    # 16 & 17
    sensor_data.append(self.ending_coords.x_val)
    sensor_data.append(self.ending_coords.y_val)

    return sensor_data

 
  def _extract_scene_image(self, sim_img_response_list):
    scene_img_response_obj = None
    for idx in range(0, len(sim_img_response_list)):
      if sim_img_response_list[idx].image_type == airsim.ImageType.Scene: 
        scene_img_response_obj = sim_img_response_list[idx]
        break
      
   # originally, the image is a 1D python list; want to turn it into a 2D numpy array
    img = np.fromstring(scene_img_response_obj.image_data_uint8, dtype=np.uint8).reshape(scene_img_response_obj.height,
                                                                                                                            scene_img_response_obj.width,
                                                                                                                            4)  # 4th channel is alpha
    print(img.shape)  # debug

    # chop off unwanted top and bottom of image (mostly deadspace or where too much white)
    img = img[self.first_scene_row_idx : self.last_scene_row_idx]

    # make grayscale since not need faster training more so than colors for now
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # not instant to do, but should help training

    # for debugging and getting cameras correct
    #cv2.imwrite('scene_{}.jpg'.format(time.time()), img)
    
    return img
      

  def _extract_depth_planner_image(self, sim_img_response_list):
    """
    Extract depth planner img from response obj in ^ and apply self.PHI to a depth planner image.
    
    :param sim...: list of image responses w/ 1 ImageResponse object for depth_planner
    that has its data that can be reshaped into a 2D (i.e., 1 channel) array / an image
    
    :returns: 2D numpy array of float32; preprocessed depth planner image
    """
    
    depth_planner_img_response_obj = None
    for idx in range(0, len(sim_img_response_list)):
      if sim_img_response_list[idx].image_type == airsim.ImageType.DepthPlanner: 
        depth_planner_img_response_obj = sim_img_response_list[idx]
        break
      
   # originally, the image is a 1D python list; want to turn it into a 2D numpy array
    img = airsim.list_to_2d_float_array(depth_planner_img_response_obj.image_data_float,
                                                   depth_planner_img_response_obj.width, depth_planner_img_response_obj.height)
    # chop off unwanted top and bottom of image (mostly deadspace or where too much white)

    print(img.shape)  # debug
    img = img[self.first_depth_planner_row_idx : self.last_depth_planner_row_idx]

    # apply PHI to each pixel - can only do if 2 dimension, i.e. grayscale

    # note: could leave as 1d array cutoff frac_top_to_drop*height first many cols and then do multiprocessing?
    if self.PHI is not None:   # could leave a None and just skip this part (might be good idea since NN not care if look nice?)
      for row_idx in range(0, img.shape[0]):
        for col_idx in  range(0, img.shape[1]):
          img[row_idx][col_idx] = self.PHI(img[row_idx][col_idx])

    # for debugging and getting cameras correct
    #cv2.imwrite('depthPlanner_{}.jpg'.format(time.time()), img)
    
    return img


  def _request_sim_images(self, scene=True, depth_planner=True):
    """
    Helper to get_composite_sim_image. Make a request to the simulation from the
    client for a snapshot from each of the 4 cameras.

    Note: only supports what i've hard coded it to do: get scene and depth planner

    :param _: if true, will request that image

    :returns: list where each element is an airsim.ImageResponse() ... 1st is scene, 2nd is depth_planner
    """
    if scene == True and depth_planner == True:
      return self.client.simGetImages(self.list_of_img_request_objects)
    
    else:
      print('ERROR: must request both scene and depth planner; no support for otherwise')
      return None


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


  def _manhattan_distance(self, x0, x1, y0, y1):
    return abs(x0 - x1) + abs(y0 - y1)


  def _euclidean_distance(self, x0, x1, y0, y1):
    return math.sqrt( (x1 - x0)**2 + (y1 - y0)**2 )


  def _arrived_at_destination(self, car_info):
    """
    Check if a car has arrived at the destination by checking if the car
    is within the end zone, which is a circle. Check if distance from end
    zone is < radius

    :param car_info: get by getCarState() via airsim client
    """
    current_distance_from_destination = self._euclidean_distance(car_info.kinematics_estimated.position.x_val,
                                                                            self.ending_coords.x_val,
                                                                            car_info.kinematics_estimated.position.y_val,
                                                                            self.ending_coords.y_val)
                                                         
    if current_distance_from_destination < self.ending_circle_radius:
      return True
    else:
      return False


  def _normalized_2_tuple(self, tup):
    magnitude_of_2_tup = math.sqrt(tup[0]**2 + tup[1]**2)
    if magnitude_of_2_tup != 0:
     # v / ||v||
      return  (tup[0] / magnitude_of_2_tup, tup[1] / magnitude_of_2_tup)
    else:
      return tup

  def _relative_bearing(self, car_yaw, car_position, destination_position):
    """
    Angle between trajectory of car and the destination. Would be 0 if
    car is headed in exact direction of destination. Is + if trajectory has the
    car going to the right of the destination; is - if to the left

    :param car_yaw: airsim yaw; if heading west in any way will be -; + is east
    :param _position: (x, y, z) airsim.Vector
    """
    car_to_dest_vector = (destination_position[0] - car_position[0], destination_position[1] - car_position[1])
  
    # calculate vector from angle and magnitude
  
    # if car yaw is negative in [-pi, 0]; convert to a positive angle [pi, 2pi]
    if car_yaw < 0:
      car_yaw = 2.0 * math.pi + car_yaw

    # use x = |v|cos(theta) and y = |v|sin(theta);
    # note that |v| is arbitrary so leave as 1
    car_heading_vector = (math.cos(car_yaw), math.sin(car_yaw))

    # normalize because will rotate car_to_dest_vector by relative_bearing to see
    # if end up @ car_heading
    car_to_dest_vector = self._normalized_2_tuple(car_to_dest_vector)
    car_heading_vector = self._normalized_2_tuple(car_heading_vector)

    # cos(theta) = (u dot v) / (||u|| ||v||)
    # relative bearing = arccos((u dot v) / (||u|| ||v||))
    # let u = car_heading_vector and let v = car_to_dest_vector
    u_dot_v = (car_heading_vector[0] * car_to_dest_vector[0]) + \
                  (car_heading_vector[1] * car_to_dest_vector[1])
  
    # only need below if have non1 |v|
    #magnitude_of_u = math.sqrt(car_heading_vector[0]**2 + car_heading_vector[1]**2)  # since |v| = 1.0
    magnitude_of_u = 1.0
    magnitude_of_v = math.sqrt(car_to_dest_vector[0]**2 + car_to_dest_vector[1]**2)

    # avoid div by 0 errors?
    if magnitude_of_u * magnitude_of_v == 0.0:
      return 0.0

    relative_bearing = math.acos(u_dot_v / (magnitude_of_u * magnitude_of_v))

    # want - if heading left of destination and + for right, so if
    # so: if you rotate car_to_dest_vector relative_bearing radians to the counter-clockwise
    # and get car_heading_vector, then car_heading_vector is to the left of
    # car_to_dest_vector; else, car_heading_vector is to the right of car_to_dest_vector
    # how to rotate a vector: https://stackoverflow.com/questions/14607640/rotating-a-vector-in-3d-space
    # thanks abstract algebra!
    x = car_to_dest_vector[0]
    y = car_to_dest_vector[1]
    theta = relative_bearing
    rotated_car_to_dest_vector = (x * math.cos(theta) - y * math.sin(theta),
                                  x * math.sin(theta) + y * math.cos(theta))
    # don't need to normalize ^ since car_to_dest_vector is already normalized

    # allowing for a little floating point error, did rotating left/counter-clockwise
    # result in car_heading_vector?
    # if yes, then negate relative_bearing; note, car_heading_vector is normalized, so no magnitude issues
    if abs(rotated_car_to_dest_vector[0] - car_heading_vector[0]) < 0.0001 and \
       abs(rotated_car_to_dest_vector[1] - car_heading_vector[1]) < 0.0001:
      relative_bearing = -1.0 * relative_bearing
    # else do nothing since relative_bearing is already >= 0

    return relative_bearing

    
  def _make_composite_image_from_responses(self, sim_img_responses):
    """
                  ---   NO LONGER IN USE   ---
    
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
