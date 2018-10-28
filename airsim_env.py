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
    self.reward_delay = 0.05
    self.episode_length = 120 # seconds
    
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
    self.emergency_reset_poses = [airsim.Pose(airsim.Vector3r(46,-530,-.7),
                                              airsim.Quaternionr(0,0,.01,1.02)),
                                  airsim.Pose(airsim.Vector3r(320,-18,-.7),
                                              airsim.Quaternionr(0,0,1.0,.01)),
                                  airsim.Pose(airsim.Vector3r(229,-313,-.7),
                                              airsim.Quaternionr(0,0,-.73,.69)),
                                  airsim.Pose(airsim.Vector3r(-800,-4,-.7),
                                              airsim.Quaternionr(0,0,.02,1.0))]

    # quaternionr is x, y, z, w for whatever reason
    # (note:sim displays w, x, y, z when press ';')
    self.normal_reset_poses = [airsim.Pose(airsim.Vector3r(63,50,-.7),
                                           airsim.Quaternionr(0,0,.71,.7)),
                               airsim.Pose(airsim.Vector3r(62,140,-.7),
                                           airsim.Quaternionr(0,0,.71,.7,)),
                               airsim.Pose(airsim.Vector3r(114,228,-.7),
                                           airsim.Quaternionr(0,0,-0.37,.99)),
                               airsim.Pose(airsim.Vector3r(296,51,-.7),
                                           airsim.Quaternionr(0,0,-.0675,.738)),
                               airsim.Pose(airsim.Vector3r(298, 23, -.7),
                                           airsim.Quaternionr(0,0,-.653,.721)),
                               airsim.Pose(airsim.Vector3r(342,-14,-.7),
                                           airsim.Quaternionr(0,0,-1.06,.014)),
                               airsim.Pose(airsim.Vector3r(200,10.4,-.7),
                                           airsim.Quaternionr(0,0,1.0,.009)),
                               airsim.Pose(airsim.Vector3r(166,-65,-.7),
                                           airsim.Quaternionr(0,0,-.661,0.75)),
                               airsim.Pose(airsim.Vector3r(230,304,-.7),
                                           airsim.Quaternionr(0,0,-.708,.711)),
                               airsim.Pose(airsim.Vector3r(241,-481,-.7),
                                           airsim.Quaternionr(0,0,1.006,0)),
                               airsim.Pose(airsim.Vector3r(64,-520,-.7),
                                           airsim.Quaternionr(0,0,.712,.707)),
                               airsim.Pose(airsim.Vector3r(-24,-529,-.7),
                                           airsim.Quaternionr(0,0,.004,1.0)),
                               airsim.Pose(airsim.Vector3r(-5.5,-300,-.7),
                                           airsim.Quaternionr(0,0,.7,.7,)),
                               airsim.Pose(airsim.Vector3r(-236,-11,-.7),
                                           airsim.Quaternionr(0,0,1.0,0)),
                               airsim.Pose(airsim.Vector3r(-356,94,-.7),
                                           airsim.Quaternionr(0,0,-1.0,0)),
                               airsim.Pose(airsim.Vector3r(-456,-11,-.7),
                                           airsim.Quaternionr(0,0,1.0,.007)),
                               airsim.Pose(airsim.Vector3r(-553,22.5,-.7),
                                           airsim.Quaternionr(0,0,.702,.712)),
                               airsim.Pose(airsim.Vector3r(-661,148.3,-.7),
                                           airsim.Quaternionr(0,0,-.03,1.0)),
                               airsim.Pose(airsim.Vector3r(-480,241,-.7),
                                           airsim.Quaternionr(0,0,-.07,1.0)),
                               airsim.Pose(airsim.Vector3r(-165,85,-.7),
                                           airsim.Quaternionr(0,0,-.687,.72)),
                               airsim.Pose(airsim.Vector3r(89,89,-.7),
                                           airsim.Quaternionr(0,0,.01,1.0))]
    self.episode_start_time = time.time()

  def step(self, action):
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
    
    done = False
    if (time.time() - self.episode_start_time) > self.episode_length:
      done = True
      
    return state_t2, reward, done, {}

  def reset(self):
    self.client.simPause(True)
    self.client.armDisarm(True)
    self.client.reset()
    emergency_pose = self.emergency_reset_poses[random.randint(0, len(self.emergency_reset_poses)-1)]
    
    self.client.simSetVehiclePose(pose=emergency_pose,
                                  ignore_collison=True)
    
    self.episode_state_time = time.time()
    self.client.simPause(False)
    
    return self._get_environment_state()

  def render(self, mode='human'):
    pass
    

  def _get_reward(self, collision_info):
    if collision_info.has_collided:
      return -1
    else:
      return 1
    
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

    
