import airsim
import numpy as np
import math


class AirSimILDataCollectorAPI:
  """
  Note: this is not an event loop; it is meant to be used witin an event loop.

  A class for collection data from AirSim for imitation learning
  purposes. The idea is that it connects to the simulator, starts the
  car in some hard-coded location on the map, and then, via the terminal,
  [via inpu()] a human will choose the steerign angles for the car by
  picking a number between 0 and num_steering_angles-1. The sim pauses each time
  the human is prompted to take make a decision.

  Everytime a steering angle is selected, the corresponding simulation images
  will be written to a file and the corresponding simulation misc. data will be
  saved in a numpy array that will be saved in a .npy file once training is
  complete.

  Unlike AirSimEnv, this does not drive the car, that's up to the user. Also,
  there is no "checking for collisions" or anything like that; we just assume
  that the human will drive correctly [assumes valid data]. Also also, we don't
  do any preprocessing of the data. We just request it and write it to a file.
  """
  def __init__(self, num_steering_angles=7):
    self.num_steering_angles = num_steering_angles
    self.steering_angles = np.linspace(-1.0, 1.0, self.num_steering_angles)


    self.client = airsim.CarClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)

    # same as from AirSimEvn
    self.car_controls = airsim.CarControls(throttle=0.45, steering=0.0,
                                           is_manual_gear=True, manual_gear=1)  # should constrain speed to < 18ish mph

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'

    self._setup_my_cameras()

    # some AirSim camera setup
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

    # units are meters
    self.beginning_coords = airsim.Pose(airsim.Vector3r(-756.236, -172.518, -0.66),  # safe
                                        airsim.Quaternionr(0.0,0.0, -0.723, 0.691))
    self.current_coords = self.beginning_coords.position
    self.ending_coords = airsim.Vector3r(-702.551, -228.107, -1.0)  # appx 298 m from start; mostly straight
    self.ending_circle_radius = 15.0 # if car in circle w/ this radius, then car has arrived @ destination

    self.total_distance_to_destination = self._manhattan_distance(self.ending_coords.x_val,
                                                                  self.beginning_coords.position.x_val,
                                                                  self.ending_coords.y_val,
                                                                  self.beginning_coords.position.y_val)
    self.current_distance_from_destination = self.total_distance_to_destination


  def set_steering_angle(self, steering_number):
    """
    Change steering angle; steering_number corresponds to one of
    self.num_steering_angles steering angles.
    """
    steering_angle = self.steering_angles[steering_number]
    print('setting steering angle to {}'.format(steering_angle))
    self.car_controls.steering = steering_angle
    self.client.setCarControls(self.car_controls)  # tested to see if conrols changed even if set before unpause: it does


  def get_sim_data(self):
    """
    Request the data from the simulation, put it into the appropriate numpy
    array format, and write it to a file.

    NOTE: call this after set_steering_angle but before unpausing
    """
    list_of_img_response_objects = self._request_sim_images()
    car_info = self.client.getCarState()


    # perform some intermediate calculation(s)
    self.current_distance_from_destination =  self._manhattan_distance(car_info.kinematics_estimated.position.x_val,
                                                                       self.ending_coords.x_val,
                                                                        car_info.kinematics_estimated.position.y_val,
                                                                        self.ending_coords.y_val)

    return self._make_row_of_data(car_info, list_of_img_response_objects)


  def pause_sim(self):
    self.client.simPause(True)


  def unpause_sim(self):
    self.client.simPause(False)


  def reset_vehicle(self):
    """
    Note: should be called when 1st iteration of event loop starts
    """
    self.client.simSetVehiclePose(pose=self.beginning_coords, ignore_collison=True)


  def arrived_at_destination(self):
    """
    Check if a car has arrived at the destination by checking if the car
    is within the end zone, which is a circle. Check if distance from end
    zone is < radius

    :param car_info: get by getCarState() via airsim client
    """
    if self.current_distance_from_destination < self.ending_circle_radius:
      return True
    else:
      return False


  def _make_row_of_data(self, car_info, list_of_img_response_objects):
    """
    :param car_info: returned by airsim.getCarState(), so is airsim.CarState object
    :param list_of_img_response_objects: list of length 2; @idx=0 is scene airsim.ImageResponse, and @idx=1
        is airsim.ImageResponse for depth planner
    :returns: statet2 (a list in ordered as model from keras_drl_... .py expects), reward_t (which is a float [-1, 1])

    Note: assumes that  self.current_distance_from_destination has been updated already
    """
    # scene, depth, misc, y/idx of current steering angle
    return np.array([self._extract_scene_image(list_of_img_response_objects),
                     self._extract_depth_planner_image(list_of_img_response_objects),
                     self._extract_misc_data(car_info),
                     np.where(self.steering_angles==self.car_controls.steering)[0][0]])


  # FROM HERE DOWN IS MORE OR LESS COPY-PASTED FROM AirSimEnv
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


  def _extract_misc_data(self, car_info):
    """
    Misc. data as from last iteration of AirSimEnv (02/06/2019)
    """
    misc_data = np.empty(0)

    # 3 manhattan distance til end
    misc_data = np.append(misc_data, self.current_distance_from_destination)

    # 4 yaw, which is relative to the world frame, so can translatie invariant?
    yaw = airsim.to_eularian_angles(car_info.kinematics_estimated.orientation)[2]  # (pitch, roll, yaw)
    misc_data = np.append(misc_data, yaw)

    # 5 relative bearing
    bearing = self._relative_bearing(yaw, (car_info.kinematics_estimated.position.x_val, car_info.kinematics_estimated.position.y_val),
                                                            (self.ending_coords.x_val, self.ending_coords.y_val))
    #print(bearing)  # debug
    misc_data =np.append(misc_data,  bearing)

    # 6 steering angle
    misc_data =np.append(misc_data, self.car_controls.steering)

    # 13
    misc_data =np.append(misc_data, car_info.speed)

    # 14
    misc_data =np.append(misc_data, abs(self.ending_coords.x_val - car_info.kinematics_estimated.position.x_val))
    misc_data =np.append(misc_data, abs(self.ending_coords.y_val - car_info.kinematics_estimated.position.y_val))

    return misc_data


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
