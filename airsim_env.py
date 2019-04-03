"""
Based on:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""
from gym import spaces, Env
import airsim
import numpy as np
import random
import cv2
import math
import time
from proximity import ProximitySensor


class AirSimEnv(Env):
  """Keras-rl usable gym (an openai package)"""

  def __init__(self,
                  num_steering_angles,
                  depth_settings_md_size,
                  scene_settings_md_size,
                  max_num_steps_in_episode=10**4,
                  fraction_of_top_of_scene_to_drop=0.0,
                  fraction_of_bottom_of_scene_to_drop=0.0,
                  fraction_of_top_of_depth_to_drop=0.0,
                  fraction_of_bottom_of_depth_to_drop=0.0,
                  seconds_pause_between_steps=0.6,  # gives rand num generator time to work (wasn't working b4)
                  seconds_between_collision_in_sim_and_register=1.0,  # note avg 4.12 steps per IRL sec on school computer
                  lambda_function_to_apply_to_depth_pixels=None,
                  need_channel_dimension=False,
                  proximity_instead_of_depth_planner=False,
                  concat_x_y_coords_to_channel_dim=False,
                  convert_scene_to_grayscale=False,
                  want_depth_image=True,
                  train_frequency=4):
    """
    Note: preprocessing_lambda_function_to_apply_to_pixels is applied to each pixel,
    and the looping through the image is handled by this class. Therefore, only 1
    parameter to this lambda function, call it pixel_value or something. Default is
    a do nothing function.
    """
    self.train_frequency = train_frequency
    
    self.want_depth_image = want_depth_image
    if proximity_instead_of_depth_planner:
      self.want_depth_image = True
      
    # 1st 2 must reflect settings.json
    self.convert_scene_to_grayscale = convert_scene_to_grayscale
    if convert_scene_to_grayscale:
      self.SCENE_INPUT_SHAPE = (scene_settings_md_size[0], scene_settings_md_size[1]*3)  
    else:
      self.SCENE_INPUT_SHAPE = (scene_settings_md_size[0], scene_settings_md_size[1]*3, 3)  
      
    self.DEPTH_PLANNER_INPUT_SHAPE = (depth_settings_md_size[0], depth_settings_md_size[1]*3) 
    self.SENSOR_INPUT_SHAPE = (7,)

    self.PROXIMITY_INPUT_SHAPE = (0,)
    self.proximity_instead_of_depth_planner = proximity_instead_of_depth_planner
    if self.proximity_instead_of_depth_planner is True:
      self.proximity_sensor = ProximitySensor(max_distance=13.0, kill_distance=3.0, num_proximity_sectors=16)
      self.PROXIMITY_INPUT_SHAPE = (self.proximity_sensor.num_proximity_sectors,)

    self.need_channel_dimension = need_channel_dimension
    self.concat_x_y_coords_to_channel_dim  = concat_x_y_coords_to_channel_dim

    if self.concat_x_y_coords_to_channel_dim is True:
      assert self.need_channel_dimension == True

    # sim admin stuff
    self.seconds_pause_between_steps = seconds_pause_between_steps
    self.seconds_between_collision_in_sim_and_register = seconds_between_collision_in_sim_and_register

    # image stuff
    if lambda_function_to_apply_to_depth_pixels is None:  # PHI from DQN algorithm
      self.PHI = None  # PHI from DQN algorithm
    else:
      self.PHI =  np.vectorize(lambda_function_to_apply_to_depth_pixels)  # 10x faster than for-looping through pixels


    self.first_scene_row_idx = int(self.SCENE_INPUT_SHAPE[0] * fraction_of_top_of_scene_to_drop)
    self.last_scene_row_idx = int(self.SCENE_INPUT_SHAPE[0] * (1-fraction_of_bottom_of_scene_to_drop))

    self.first_depth_planner_row_idx = int(self.DEPTH_PLANNER_INPUT_SHAPE[0] * fraction_of_top_of_depth_to_drop)
    self.last_depth_planner_row_idx = int(self.DEPTH_PLANNER_INPUT_SHAPE[0] * (1-fraction_of_bottom_of_depth_to_drop))

    #print('frac bottom depth', fraction_of_bottom_of_depth_to_drop)  # debug
    #print('1st row idx scene', self.first_scene_row_idx)  # debug
    #print('last row idx depth', self.last_depth_planner_row_idx)   # debug

    assert self.first_scene_row_idx < self.last_scene_row_idx
    assert self.first_depth_planner_row_idx < self.last_depth_planner_row_idx


    # note the shapes of inputs to the neural network; can retrieve in keras_drl_steering... .py
    self.SCENE_INPUT_SHAPE = (self.last_scene_row_idx-self.first_scene_row_idx, self.SCENE_INPUT_SHAPE[1])
    self.DEPTH_PLANNER_INPUT_SHAPE = (self.last_depth_planner_row_idx-self.first_depth_planner_row_idx, self.DEPTH_PLANNER_INPUT_SHAPE[1])
    self.SENSOR_INPUT_SHAPE = self.SENSOR_INPUT_SHAPE


    # steering stuff
    self.action_space = spaces.Discrete(num_steering_angles)
    self.action_space_steering = np.linspace(-1.0, 1.0, num_steering_angles).tolist()
    self.car_controls = airsim.CarControls(throttle=0.6,
                                                       steering=0.0,
                                                       is_manual_gear=True,
                                                       manual_gear=1)  # should constrain speed to < 18ish mph
    # m/s = mph x 0.44704
    self.max_meters_per_second = 35.0 * 0.44704  # 35.0 mph is reasonable max for that windy road
    # in-sim episode handling
    self.episode_step_count = 0
    self.total_step_count = 0
    self.steps_per_episode = max_num_steps_in_episode
    self.total_num_episodes = 0  # over all eisodes
    self.total_num_steps = 0   # over all episodes

    # collision info for emergency resets and reward func calc
    self.collisions_in_a_row = 0
    self.max_acceptable_collisions_in_a_row = 6 # note: if stuck, then collisions will keep piling on
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

    self.depth_planner_dilation_kernel = np.ones((3,5),np.uint8) * 255

    # requests also returned in this order; order by how concatentate img, from L to R
    if self.want_depth_image:
      self.list_of_img_request_objects = [airsim.ImageRequest(camera_name=self.left_cam_name,
                                                              image_type=airsim.ImageType.Scene,
                                                              pixels_as_float=False,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.forward_cam_name,
                                                              image_type=airsim.ImageType.Scene,
                                                              pixels_as_float=False,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.right_cam_name,
                                                              image_type=airsim.ImageType.Scene,
                                                              pixels_as_float=False,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.left_cam_name,
                                                              image_type=airsim.ImageType.DepthPlanner,
                                                              pixels_as_float=True,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.forward_cam_name,
                                                              image_type=airsim.ImageType.DepthPlanner,
                                                              pixels_as_float=True,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.right_cam_name,
                                                              image_type=airsim.ImageType.DepthPlanner,
                                                              pixels_as_float=True,
                                                              compress=False) ]
    else:
      self.list_of_img_request_objects = [airsim.ImageRequest(camera_name=self.left_cam_name,
                                                              image_type=airsim.ImageType.Scene,
                                                              pixels_as_float=False,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.forward_cam_name,
                                                              image_type=airsim.ImageType.Scene,
                                                              pixels_as_float=False,
                                                              compress=False),
                                          airsim.ImageRequest(camera_name=self.right_cam_name,
                                                              image_type=airsim.ImageType.Scene,
                                                              pixels_as_float=False,
                                                              compress=False)]


    # ; ordering: 1 2 3 4 in sim window while Quaternionr() has 2 3 4 1 for some reason
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
    """
    self.beginning_coords = airsim.Pose(airsim.Vector3r(12.314, -31.069, -0.93),  # safe
                                                  airsim.Quaternionr(-0.004,0.008, 0.236, 0.972))
    self.ending_coords = airsim.Vector3r(122.288, 12.283, -1.0)  # appx 298 m from start; mostly straight
    """

    self.beginning_coords = airsim.Pose(airsim.Vector3r(-758.236, -175.518, -0.66),  # safe
                                                  airsim.Quaternionr(0.0,0.0, -0.723, 0.691))
    #self.ending_coords = airsim.Vector3r(109.694, -396.685, -1.0)  # appx 298 m from start; mostly straight
    self.ending_coords = airsim.Vector3r(-702.551, -228.107, -1.0)  # appx 298 m from start; mostly straight

    # units are meters
    self.ending_circle_radius = 10.0 # if car in circle w/ this radius, then car has arrived @ destination

    

    # stuff used in reward function
    self.total_distance_to_destination = self._manhattan_distance(self.ending_coords.x_val,
                                                                                         self.beginning_coords.position.x_val ,
                                                                                         self.ending_coords.y_val,
                                                                                         self.beginning_coords.position.y_val)
    self.current_distance_from_destination = self.total_distance_to_destination
    self.previous_coords = self.beginning_coords.position
    self.elapsed_episode_time_in_simulation_secs = 1.0   # tracks time between unpause and pause in step; used
    self.current_distance_travelled_towards_destination = 0.0
    self.episode_time_in_irl_seconds = 0.0  # only used for debug and steps/second measurement,
    # not for tracking progress in sim; note: sim steps per real life seconds is ~6
    self.sim_is_paused = False

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
    self.client.setCarControls(self.car_controls)  # tested to see if conrols changed even if set before unpause: it does
  
    sim_unpaused_start_time = time.time()  # used to track time at which sim is unpaused

    self.client.simPause(False)
    time.sleep(self.seconds_pause_between_steps)
    self.client.simPause(True)

    self.episode_step_count += 1
    self.total_step_count +=  1


    self.elapsed_episode_time_in_simulation_secs += time.time() - sim_unpaused_start_time
    # END TIMESTEP
    
    # can, in fact, get this information once pause, so get it now
    list_of_img_response_objects = self._request_sim_images()
    collision_info = self.client.simGetCollisionInfo()
    car_info = self.client.getCarState()

    # END TIMESTEP
    # BEGIN BACKEND   # reward_t and state_t2

    # used in reward and state making
    # track collisions in a row for checking if stuck [e.g., on a curb]
    #print(collision_info.has_collided, collision_info.object_id)  # debug
    #print(collision_info.object_name)   # debug
    if collision_info.object_name is not '' and car_info.speed < 1.0:
      if self.obj_id_of_last_collision == collision_info.object_id:
        self.collisions_in_a_row += 1
      else:
        self.collisions_in_a_row = 1
    if  car_info.kinematics_estimated.position.z_val < -0.698:
      self.collisions_in_a_row = self.max_acceptable_collisions_in_a_row - 1
      self.obj_id_of_last_collision = collision_info.object_id

    
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

    state_t2 = self._make_state(car_info, list_of_img_response_objects)

    done = self._is_done(car_info)

    # mostly for debug, but can be helpful
    if self.episode_step_count % 30 == 0:
      print('Ep step {}, averaging {} steps per IRL sec'.format(self.episode_step_count,
                                                                                  (self.episode_step_count / (time.time() -self.episode_time_in_irl_seconds))))
      #print('\t...averaging {} steps per sim sec (assuming x1.0 speed).'.format(self.episode_step_count / (self.elapsed_episode_time_in_simulation_secs)))
    # check if made it to w/in the radius of the destination area/circle
    if self.episode_step_count % 11 == 0:
      print(reward_t)

    self.previous_coords = car_info.kinematics_estimated.position

    return state_t2, reward_t, done, {}


  def reset(self):
    """
    When done (in step()) is true, this function is called.

    :returns: what the cartpole example returns (standard? for openai gym env.reset())
    """

    self.client.simPause(True)
    time.sleep(1)

    print('So far: {} steps over {} episodes'.format(self.total_num_steps, self.total_num_episodes))

    self.total_num_episodes += 1
    self.total_num_steps += self.episode_step_count


    self.episode_step_count = 0
    self.collisions_in_a_row = 0
    self.obj_id_of_last_collision = -123456789 # any int < -1 is ok

    self.elapsed_episode_time_in_simulation_secs = 0.1  # avoid div by 0 err when call _make_state @ fin

    self.client.simPause(False)
    self.client.reset()

    # need next 2 because: https://microsoft.github.io/AirSim/docs/apis/
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    #reset_pose = self.reset_poses[random.randint(0, len(self.reset_poses)-1)]  # from when would drive aimlessly

    self.client.simSetVehiclePose(pose=self.beginning_coords,
                                           ignore_collison=True)
    self.previous_coords = self.beginning_coords.position
    self.episode_time_in_irl_seconds = time.time()  # again, for debug purposes
    self.current_distance_from_destination = self.total_distance_to_destination
    self.elapsed_episode_time_in_simulation_secs = 1.0

    time.sleep(4)  # crash issue with requesting stuff before actually capable of resettting? idk

    list_of_img_response_objects = self._request_sim_images()
    car_info = self.client.getCarState()

    # initial state for this new episode
    return self._make_state(car_info, list_of_img_response_objects)


  def render(self, mode='human'):
    pass  # airsim server binary handles rendering; we're just the client


  def _is_done(self, car_info):
      """
      Returns True done if episode timer runs out (1) OR if fallen into oblilvion (2)
      # OR if spinning out of control (3) OR if knocked into the stratosphere (4)
      OR arrived at destination (end of road)
      """
      done = False
      if self.episode_step_count >  self.steps_per_episode or \
         car_info.kinematics_estimated.position.z_val > -0.58 or \
         abs(car_info.kinematics_estimated.orientation.y_val) > 0.3125 or \
         car_info.speed > 17.0 or \
         self.collisions_in_a_row > self.max_acceptable_collisions_in_a_row or \
         car_info.kinematics_estimated.position.z_val < -2.0: # if on sidewalk
        done = True

      if self._arrived_at_destination(car_info):
        done = True
        
      return done

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

    # if have made very little progress towards goal so far -- note, get about 3 to 4 steps per IRL sec on school computer
    if (self.total_distance_to_destination - self.current_distance_from_destination) <  15.0 and self.episode_step_count > 100:
      reward = -1.0

    elif car_info.kinematics_estimated.position.z_val < -0.695 or car_info.kinematics_estimated.position.z_val > -0.625:
      reward = -1.0

    elif sec_since_last_collision < self.seconds_between_collision_in_sim_and_register and collision_info.time_stamp != 0.0:  # irl seconds
      self.distance_since_last_collision = 0.0
      reward = -1.0

    # if hit curb or an unnamed object
    elif (collision_info.object_id == -1 and collision_info.object_name != '' and car_info.speed < 1.0) or \
         (abs(car_info.kinematics_estimated.orientation.x_val) > 0.035 or abs(car_info.kinematics_estimated.orientation.y_val) > 0.035):   # check if hit curb (car x and y orientation changes)
      self.distance_since_last_collision = 0.0
      reward = -0.1

    else:

      # will always be >= 0
      distance_reward =  self.current_distance_travelled_towards_destination / self.total_distance_to_destination

      #reward = time_reward + distance_reward
      reward = max(0, distance_reward - 0.03)  # don't reward until get sufficiently far out - should help avoid driving in circles

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
    scene_image = self._extract_scene_image(list_of_img_response_objects)
    
    state_t2.append(self._transform_scene_image(scene_image))

    if self.want_depth_image:
      depth_planner_image = self._extract_depth_planner_image(list_of_img_response_objects)
      
      if self.proximity_instead_of_depth_planner is True:
        proximities_by_sector = np.array(self.proximity_sensor.depth_planner_image_to_proximity_list(depth_planner_image))
        if self.need_channel_dimension == True:  # channels need own dimension for Conv3D and Conv2DRnn
          proximities_by_sector =  proximities_by_sector.reshape(proximities_by_sector.shape[0], 1)
        state_t2.append(proximities_by_sector)
        
      else:
        state_t2.append(self._transform_depth_planner_image(depth_planner_image))
                      
    state_t2.append(self._extract_sensor_data(car_info))
    #print('shape of state', np.array(state_t2).shape)  # for debug
    return state_t2  # order should be: scene img, depth img, sensor data


  def _extract_sensor_data(self, car_info):
    """
    Returns a list w/ 17 entries:
    # all x's were included at one point, but now are either unreliable or I think they are worthless
    x-x. GPS (x, y) coordinates of car
    3. manhattan distance from end point (x, y)
    4. yaw/compass direction of car in radians  # use airsim.to_eularian_angles() to get yaw
    5. relative bearing in radians (see func)
    6. current steering angle in [-1.0, 1.0]
    x-x. linear velocity (x, y) # not accurate whatsoever (press ';' in sim to see)
    x-x. angular velocity (x, y)
    x-x. linear acceleration (x, y)
    x-x. angular acceleration (x, y)
    13. speed
    14-15. absolute difference from current location to destination for x and y each
    x-x. (x, y) coordinates of destination

    Note: no z coords because city is almost entirely flat
    """
    sensor_data = np.empty(0)

    # 1-2 car's gps x and y coords
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.position.x_val)
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.position.y_val)

    # 3 manhattan distance til end
    sensor_data =np.append(sensor_data, self.current_distance_from_destination)

    # 4 yaw, which is relative to the world frame, so can translatie invariant?
    yaw = airsim.to_eularian_angles(car_info.kinematics_estimated.orientation)[2]  # (pitch, roll, yaw)
    sensor_data =np.append(sensor_data, yaw)

    # 5 relative bearing
    bearing = self._relative_bearing(yaw, (car_info.kinematics_estimated.position.x_val, car_info.kinematics_estimated.position.y_val),
                                                            (self.ending_coords.x_val, self.ending_coords.y_val))
    #print(bearing)  # debug
    sensor_data =np.append(sensor_data,  bearing)

    # 6 steering angle
    sensor_data =np.append(sensor_data, self.car_controls.steering)

    # 7 angular velocity x and y
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.angular_velocity.x_val)
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.angular_velocity.y_val)

    # 9
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.linear_acceleration.x_val)
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.linear_acceleration.y_val)

    # 11
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.angular_acceleration.x_val)
    #sensor_data =np.append(sensor_data, car_info.kinematics_estimated.angular_acceleration.y_val)

    # 13
    sensor_data =np.append(sensor_data, car_info.speed)

    # 14
    sensor_data =np.append(sensor_data, abs(self.ending_coords.x_val - car_info.kinematics_estimated.position.x_val))
    sensor_data =np.append(sensor_data, abs(self.ending_coords.y_val - car_info.kinematics_estimated.position.y_val))

    # 16 & 17
    #sensor_data =np.append(sensor_data, self.ending_coords.x_val)
    #sensor_data =np.append(sensor_data, self.ending_coords.y_val)

    #print('sensor data shape', sensor_data.shape)  # for debug
    if self.need_channel_dimension == True:  # channels need own dimension for Conv3D and Conv2DRnn
      return np.array(sensor_data).reshape(sensor_data.shape[0], 1)
    else:
      return np.array(sensor_data).reshape(sensor_data.shape[0])


  def _extract_scene_image(self, sim_img_response_list):
    scene_img_responses = []
    for idx in range(0, len(sim_img_response_list)):
      if sim_img_response_list[idx].image_type == airsim.ImageType.Scene:

        scene_img_responses.append(sim_img_response_list[idx])

    # originally, the image is in 3 1D python lists; want to turn them into a 2D numpy arrays
    img = np.concatenate([np.fromstring(img_response_obj.image_data_uint8,
                                        dtype=np.uint8).reshape(img_response_obj.height,
                                                                img_response_obj.width,
                                                                4) for img_response_obj in scene_img_responses],
                         axis=1)
      
    return img


  def _transform_scene_image(self, img):
    # chop off unwanted top and bottom of image (mostly deadspace or where too much white)
    if self.first_scene_row_idx != 0 or self.last_scene_row_idx != self.SCENE_INPUT_SHAPE[0]:
      img = img[self.first_scene_row_idx : self.last_scene_row_idx]
    
    # make grayscale since not need faster training more so than colors for now
    if self.convert_scene_to_grayscale:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # not instant to do, but should help training
    else:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # https://stackoverflow.com/questions/36872379/how-to-remove-4th-channel-from-png-images

    #print('scene img shape', img.shape)  # debug

    # for debugging and getting cameras correct
    #if self.episode_step_count % 7 == 0:
    #  cv2.imwrite('scene_{}.jpg'.format(time.time()), img)

    if self.need_channel_dimension == True:  # channels need own dimension for Conv3D and Conv2DRnn
      if self.convert_scene_to_grayscale:   # if need channel dimension but is in gray scale...
        img =  img.reshape(img.shape[0], img.shape[1], 1)

    if self.concat_x_y_coords_to_channel_dim is True:
      return self.concat_x_y_coords(img)
    else:
      return img


  def _extract_depth_planner_image(self, sim_img_response_list):
    """
    Extract depth planner img from response obj in ^ and apply self.PHI to a depth planner image.

    :param sim...: list of image responses w/ 1 ImageResponse object for depth_planner
    that has its data that can be reshaped into a 2D (i.e., 1 channel) array / an image

    :returns: 2D numpy array of float32; preprocessed depth planner image
    """
    depth_planner_img_responses = []
    for idx in range(0, len(sim_img_response_list)):
      if sim_img_response_list[idx].image_type == airsim.ImageType.DepthPlanner:

        depth_planner_img_responses.append(sim_img_response_list[idx])

    # originally, the image is in 3 1D python lists of strings; want to turn them into a 2D numpy arrays
    img = np.concatenate([airsim.list_to_2d_float_array(depth_planner_img_response_obj.image_data_float,
                                                        depth_planner_img_response_obj.width,
                                                        depth_planner_img_response_obj.height) \
                          for depth_planner_img_response_obj in depth_planner_img_responses],
                          axis=1)
    return img

    
  def _transform_depth_planner_image(self, img):
    # chop off unwanted top and bottom of image (mostly deadspace or where too much white)
    if self.first_depth_planner_row_idx != 0 or self.last_depth_planner_row_idx != self.DEPTH_PLANNER_INPUT_SHAPE[0]:
      img = img[self.first_depth_planner_row_idx : self.last_depth_planner_row_idx]

    
    # apply PHI to each pixel - can only do if 2 dimension, i.e. grayscale
    # note: could leave as 1d array cutoff frac_top_to_drop*height first many cols and then do multiprocessing?
    if self.PHI is not None:   # could leave a None and just skip this part (might be good idea since NN not care if look nice?)
      img = self.PHI(img) # PHI was vectorized in __init__, so this applies PHI to each pixel in
      # image approximately 10x faster than 2-nested for-loops of applying PHI

    # canny edge detection - draws white-ish outlines on objects w/ distinct edges
    # surprisingly fast - averages about 0.000579 seconds
    #img = np.array(img, np.uint8)
    #canny_edges = cv2.Canny(img,10, 150)  # np.array, min, max intensity gradient

    # dilate - stretch out the areas w/ white pixels (edges in canny_edges)
    #dilated_canny_edges = cv2.dilate(canny_edges, self.depth_planner_dilation_kernel, iterations = 1)

    # overlay the dilated edges on the depth planner image (draw object outlines)
    #img = cv2.addWeighted(img, 0.75, dilated_canny_edges, 0.25, 30.0)

    # for debugging and getting cameras correct
    #cv2.imwrite('depthPlanner_{}.jpg'.format(time.time()), img)
    #print('depth_planner img shape', img.shape)  # debug

    if self.need_channel_dimension == True:  # channels need own dimension for Conv3D and Conv2DRnn
      img = img.reshape(img.shape[0], img.shape[1], 1)

    if self.concat_x_y_coords_to_channel_dim is True:
      return self.concat_x_y_coords(img)
    else:
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
    left_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=-1.04, roll=0.0)
    forward_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=0.0, roll=0.0)
    right_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=1.04, roll=0.0)
    backward_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=0.0, roll=0.0)


    # creates a panoram-ish camera set-up
    self.client.simSetCameraOrientation(self.left_cam_name, left_cam_orientation)
    self.client.simSetCameraOrientation(self.right_cam_name, right_cam_orientation)
    self.client.simSetCameraOrientation(self.forward_cam_name, forward_cam_orientation)


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

  def concat_x_y_coords(self, img):
    """
    Concat 2 channels beind the first: one with the x coordinate of the given pixel
    and another with the y coordinate of the given pixel. Note: this is really
    only for when you can't use the Keras layer to do this.
    """
    img_h = img.shape[0]
    img_w = img.shape[1]

    # get img_w evenly spaced real numbers in interval [-1, 1], one for each column in img
    x_coords_of_any_given_pixel_in_any_given_row = np.linspace(start=-1.0, stop=1.0, num=img_w, endpoint=True)
    
    # get img_h evenly spaced real numbers in interval [-1, 1], one for each row in img
    y_coords_of_any_given_pixel_in_any_given_row = np.linspace(start=-1.0, stop=1.0, num=img_h, endpoint=True)

    if self.convert_scene_to_grayscale:
      img = np.concatenate([img,
                              x_coords_of_any_given_pixel_in_any_given_row.repeat(img_h).reshape(img.shape),
                              y_coords_of_any_given_pixel_in_any_given_row.repeat(img_w).reshape(img.shape)], axis=-1)
    else:
      img = np.concatenate([img,
                              x_coords_of_any_given_pixel_in_any_given_row.repeat(img_h).reshape((img_h, img_w, 1)),
                              y_coords_of_any_given_pixel_in_any_given_row.repeat(img_w).reshape((img_h, img_w, 1))], axis=-1)
      
    #print(img.shape)  # for debug - should see channels be 3 rather than 1
    return img


