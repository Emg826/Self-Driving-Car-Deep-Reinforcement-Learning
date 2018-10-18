"""
The purpose of this file is to create a deep reinforcement learning
system to control the steering ONLY. The car will be kept in 1st gear
to restrain the speed.

Therefore, the first goal is to get a deep reinforcement algorithm that can,
at the very least, steer itself to avoid collisions.

The second goal is to come up with a usable reward/penalty function. 

The third and last goal (for this file at least) is to just figure out how to
use multiple simulation images with a deep Q network. This means learning how
to get multiple images and how to actually implement the deep Q network.

Note*: my DQN driver agent does not stack frames like in DeepMind paper


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
from keras.layers import Convolution2D, Flatten, Dense, MaxPooling2D
from keras.regularizers import l1_l2
from keras.optimizers import RMSprop
import keras.backend as K

IMG_SHAPE = (260, 1190,  4) # H x W x NUM_CHANNELS
random.seed(3)


def dqn_loss(y_true, y_pred):
  #for debugging purposes - mse
  return K.mean(K.square(y_pred - y_true), axis=-1) #  mean of each row?

  # note: each idx in y_true or y_pred is an output, so
  # so each of these is 2D, right?
def acvasdf(y_true, y_pred):
  print('hello')
  y_true = K.eval(y_true) # dense_4_target?
  print('First one eval\'d')
  y_pred = K.eval(y_pred)
  
  total_squared_loss = 0.0
  for r_idx in range(0, y_true.shape[0]):
    print(r_idx)
    targets = y_true[r_idx]
    predictions = y_pred[r_idx]

    max_action_idx = np.argmax(target_Q) 
    target_Q = targets[max_action_idx]
    predicted_Q = predictions[max_action_idx]

    total_squared_loss += (target_Q - predicted_Q)

  print('goodbye')
  return K.variable((total_squared_loss / y_true.shape[0]), )



                               
class AirSimEnv():
  """
  This is the virtual representation of the environment with which
  the agent will interact. To the driver, abstraction allows the
  "environment" of the simulation to just be represented by an
  image and the car.
  """
  def __init__(self):
    self.client = airsim.CarClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'

    self.setup_my_cameras()
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
    return self.make_composite_image_from_responses(sim_img_responses,
                                                    cam_names)

  def get_collision_state(self):
    """ Returns False if no collision; True otherwise"""
    return self.client.simGetCollisionInfo().has_collided

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
    self.client.setCarControls(car_controls)

  def setup_my_cameras(self):
    """
    Helper function to set the left, right, forward, and back cameras up
    on the vehicle as I've see fit.

    :param client: airsim.CarClient() object that
    already connected to the sim

    :returns: nada
    """
    # pitch, roll, yaw ; each is in radians where
    # 15 degrees = 0.261799 (don't ask me why they used radians...)
    # NOTE: these images are reflected over line x = 0: left in the image
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
  
  def freeze_sim(self):
    self.client.simPause(True)

  def unfreeze_sim(self):
    self.client.simPause(False)
    
  def emergency_reset(self):
    """
    Reset at emergency reset coords and scoot emergency respawn
    coords back a little bit.
    """
    print('EMERGENCY RESET TRIGGERED!')
    self.freeze_sim()
    self.client.armDisarm(True)
    self.client.reset()  # needed to make car stop rotating upon respawn
    emergency_pose = self.emergency_reset_poses[random.randint(0, len(self.emergency_reset_poses)-1)]
    
    self.client.simSetVehiclePose(pose=emergency_pose,
                                  ignore_collison=True)

    print('\t respawned at {}'.format(emergency_pose.position))
    print('\t \t with orientation {}'.format(emergency_pose.orientation))
    
    self.unfreeze_sim()                               
                                                   
  def normal_reset(self):
    """
    Pauses the simulation before resetting the vehicle.
    """
    self.freeze_sim()
    self.client.armDisarm(True)
    self.client.reset()
    normal_pose = self.normal_reset_poses[random.randint(0, len(self.normal_reset_poses)-1)]
    self.client.simSetVehiclePose(pose=normal_pose,
                                  ignore_collison=True)
    print('\t respawned at {}'.format(normal_pose.position))
    print('\t \t with orientation {}'.format(normal_pose.orientation))
    self.unfreeze_sim()


class DriverAgent():
  """
  The driver of the vehicle, a deep Q network that uses
  deep Q learning.
  """
  def __init__(self, num_steering_angles, replay_memory_size,
               mini_batch_size=32, gamma=0.98,
               load_most_recent_weights=False,
               specific_weights_filepath=None):
    """
    Initialize the Deep Q driving agent
    
    :param num_steering_angles: number of steering angles, equidistant from each other, in [-1, 1]
    :param replay_memory_size: number of previous states to remember, sample from this many states
    :param mini_batch_size: size of a batch to randomly sample from replaymemory
    :param gamma: discount rate of next reward (and determines all future rewards)
    :param load_most_recent_weights: boolean; True if try to load most recent weights into networks
    :param specific_weights_filepath: filepath (from curr. working dir.) to .h5 file of weights
    """
    self.replay_memory = ReplayMemory(replay_memory_size=replay_memory_size,
                                      mini_batch_size=mini_batch_size)

    self.mini_batch_size = mini_batch_size
    self.gamma = gamma
    
    # airsim steering angles are in the interval [-1, 1] for some reason
    self.num_steering_angles = num_steering_angles
    self.action_space = np.linspace(-1.0, 1.0, num_steering_angles).tolist()
    # ^ is a list for because actions stored are steering angles, not ouput node idx
    # and lists have method .index(val_in_list) [used in training]

    self.optimizer = RMSprop()

    # neural network to approximate the policy/Q function
    self.online_Q = Sequential()
    self.online_Q.add(Convolution2D(128, kernel_size=8, strides=5, input_shape=IMG_SHAPE,
                                    data_format='channels_last', activation='relu',
                                    activity_regularizer=l1_l2(l1=0.01, l2=0.01)))
    self.online_Q.add(Convolution2D(92, kernel_size=5, strides=3, activation='relu',
                                    activity_regularizer=l1_l2(l1=0.01, l2=0.01)))
    self.online_Q.add(Convolution2D(64, kernel_size=3, strides=2, activation='relu',
                                    activity_regularizer=l1_l2(l1=0.01, l2=0.01)))
    self.online_Q.add(MaxPooling2D(pool_size=4, strides=2))
    self.online_Q.add(Flatten())
    self.online_Q.add(Dense(128, activation='sigmoid',
                            activity_regularizer=l1_l2(l1=0.01, l2=0.01)))  #relu = max{0, x}
    self.online_Q.add(Dense(64, activation='sigmoid',
                            activity_regularizer=l1_l2(l1=0.01, l2=0.01)))
    self.online_Q.add(Dense(32, activation='sigmoid',
                            activity_regularizer=l1_l2(l1=0.01, l2=0.01)))
    self.online_Q.add(Dense(num_steering_angles, activation='linear'))  # 1 output per steering angle

    self.offline_Q = self.online_Q # target network -- used to update weights
    self.offline_Q.compile(optimizer='adam', loss=dqn_loss)
    self.online_Q.compile(optimizer='adam', loss=dqn_loss) # used for selecting actions -- copies weights from offline

    # try to load weights if there are any to load and want to load
    self.weights_directory = os.path.join(os.getcwd(), 'deep_nn_collision_avoidance_weights')
    if specific_weights_filepath is not None:
      self.load_specific_weights_into_networks(filepath=specific_weights_filepath)
    elif load_most_recent_weights is True:
      self.load_most_recent_weights_into_networks()

    print(self.online_Q.summary())
    
  def load_specific_weights_into_networks(self, filepath):
    try:
      self.online_Q.load_weights(filepath)
      self.offline_Q.load_weights(filepath)
    except:
      print('Could not load these weights from {}. Will use random weights instead.'.format(filepath))

  def load_most_recent_weights_into_networks(self):
    """
    Load weights that were saved by .save_weights() function
    that Keras has. This should be a .h5 file located in the directory
    self.weights_directory.
    """
    # make weights directory if not already there
    if os.path.exists(self.weights_directory) is False:
      os.mkdir(self.weights_directory)
      print('Error: {} does not exist, so there are no weights to load.\n'
            'Just made this directory now.\n'
            'Using randomly initialized weights instead.\n'
            'Will save this run\'s weights to this directory for next time.\n'.format(self.weights_directory))

    # else directory exists
    else:
      # check if there are any files to load
      list_of_files = os.listdir(self.weights_directory)
        
      if len(list_of_files) == 0:
        print('No weight files to load in {}'.format(self.weights_directory))
        return
      else:
        list_of_files.sort()
        file_name_of_weights = list_of_files[-1] # expect .h5 file

        try:
          self.online_Q.load_weights(os.path.join(self.weights_directory,
                                                  file_name_of_weights))
          self.offline_Q.load_weights(os.path.join(self.weights_directory,
                                                   file_name_of_weights))
        except:
          print('Could not load weights. Perhaps the network architectures do not match?')

    

  def current_reward(self, car_state, just_collided):
    """
    Custom reward function

    :param car_state: airsim.CarState() of car in sim
    
    :returns: the current reward given the state of the car

    *Note: want to keep the rewards in [-1, 1] (see DeepMind Nature paper)
    """
    if just_collided is True:
      # collisions attrib has object_id (collided w/), so
      # very easily could modify and see if collided with person
      # or vehicle or building or curb or etc.
      return -1.0
    else:
      return 0.01

  def my_train(self):
    mini_batch = self.replay_memory.mini_batch_random_sample()

    s = [quadruple[0] for quadruple in mini_batch]
    a = [quadruple[1] for quadruple in mini_batch]
    r = [quadruple[2] for quadruple in mini_batch]
    s2 = [quadruple[3] for quadruple in mini_batch]
    
    # delta = r + (1-terminal) * gamma * max_a Q^(s2, a) - Q(s, a)
    # list of terminal/not terminal step
    term = [1 if s2 is None else 0 for state in s2] # not what they do, but get same result later

    # need to do this, else will give Q net None objects if have terminal quadruples
    for idx, elem in enumerate(s2):
      if elem is None:
        s2[idx] = s[idx]
        
    # get max q values from offline predictions: -- Compute max_a Q(s_2, a).

    batch_of_q_vectors = self.offline_Q.predict(np.asarray(s2))
    q2_max = [np.max(q_vector) for q_vector in batch_of_q_vectors] # action decided in past, so don't care here

    # -- Compute q2 = (1-terminal) * gamma * max_a Q^(s2, a)
    q2 = [(1-term_val)*self.gamma * q_val for q_val, term_val in zip(q2_max, term)]

    delta = r[:]

    # rescale r's if you want to, but i won't
    delta = [r1_val + q2_val for r1_val, q2_val in zip(r, q2)]
    # ^ is now just the list of targets?

    # now get the online Q's, the predicteds
    # q = Q(s, a)
    q_all = self.online_Q.predict(np.asarray(s)) # mini_batch_size x num_steering angles
    q = [0.0] * self.mini_batch_size # mini_batch_size x 1
    for i_th_timestep in range(len(q_all)):  # note: i think lua uses 1 as 1st idx
      # get just predicted q val for action that was chosen in past
      # self.action_space.index(a[i_th_timestep]) returns an idx (0 to num_steering_angels-1)
      q[i_th_timestep] = q_all[i_th_timestep][a[i_th_timestep][0]]

    # now list of (r + (1-terminal) * gamma * max_a Q^(s2, a) - Q(s, a)^2
    # this is the loss from the DQN Nature paper
    losses = [(target - predicted)**2 for target, predicted in zip(delta, q)]

    # if i understand this correctly, this does gradient descent on loss
    # w/ respect to weights of network
    new_weights = self.optimizer.get_updates(loss=losses, params=self.online_Q.get_config())
    print(new_weights.shape)
    self.online_Q.set_weights(new_weights)


  def lua_and_keras_train(self):
    """
    This is my attempt to combine python's Keras and the lua code
    from DeepMind's github that contains the deep q network code.
    The hard part of that code is manually applying the gradients,
    but I think Keras handles that when fit() is called.
    """
    mini_batch = self.replay_memory.mini_batch_random_sample()

    s = [quadruple[0] for quadruple in mini_batch]
    a = [quadruple[1] for quadruple in mini_batch]
    r = [quadruple[2] for quadruple in mini_batch]
    s2 = [quadruple[3] for quadruple in mini_batch]
    
    # delta = r + (1-terminal) * gamma * max_a Q^(s2, a) - Q(s, a)
    # list of terminal/not terminal step
    term = [1 if s2 is None else 0 for state in s2] # not what they do, but get same result later

    # need to do this, else will give Q net None objects if have terminal quadruples
    for idx, elem in enumerate(s2):
      if elem is None:
        s2[idx] = s[idx]
        
    # get max q values from offline predictions: -- Compute max_a Q(s_2, a).

    batch_of_q_vectors = self.offline_Q.predict(np.asarray(s2))
    q2_max = [np.max(q_vector) for q_vector in batch_of_q_vectors] # action decided in past, so don't care here

    # -- Compute q2 = (1-terminal) * gamma * max_a Q^(s2, a)
    q2 = [(1-term_val)*self.gamma * q_val for q_val, term_val in zip(q2_max, term)]

    delta = r[:]

    # rescale r's if you want to, but i won't
    delta = [r1_val + q2_val for r1_val, q2_val in zip(r, q2)]
    # ^ is now just the list of targets?

    # now get the online Q's, the predicteds
    # q = Q(s, a)
    q_all = self.online_Q.predict(np.asarray(s)) # mini_batch_size x num_steering angles
    q = [0.0] * self.mini_batch_size # mini_batch_size x 1
    for i_th_timestep in range(len(q_all)):  # note: i think lua uses 1 as 1st idx
      # get just predicted q val for action that was chosen in past
      # self.action_space.index(a[i_th_timestep]) returns an idx (0 to num_steering_angels-1)
      q[i_th_timestep] = q_all[i_th_timestep][a[i_th_timestep][0]]

    # now list of r + (1-terminal) * gamma * max_a Q^(s2, a) - Q(s, a) 
    delta = [target - predicted for target, predicted in zip(delta, q)]

    # can clip delta between -1 and 1 here; i'm not going to, for now.

    # every action's q val (aside from that of action taken) is 0 since didn't do
    targets = np.zeros((self.mini_batch_size, self.num_steering_angles), dtype=np.float)
    
    for i_th_timestep in range(self.mini_batch_size):
      targets[i_th_timestep][a[i_th_timestep][0]] = delta[i_th_timestep]

    # all of the above was getQUpdates() in NeuralQLearner.lua
    self.online_Q.train_on_batch(x=np.asarray(s), y=targets)

  def copy_online_weights_to_offline(self):
    """
    Copy online network's weights to offline network. Occurs every
    C time steps (C is not an attribute of this class [though it could be later])
    """
    self.offline_Q.set_weights(self.online_Q.get_weights())

  def save_weights(self):
    """
    Save the model weights to h5 file that can be loaded in to a model
    with the exact same architecture model should the program crash
    """
    time_stamp = int(time.time())
    weights_filename = '{}.h5'.format(time_stamp)
    self.offline_Q.save_weights(os.path.join(self.weights_directory,
                                                 weights_filename))
    print('Weights saved at {}s in file {}'.format(time_stamp, weights_filename))


  def get_random_steering_angle(self):
    action_idx = random.randint(0, self.num_steering_angles-1)
    return (action_idx, self.action_space[action_idx])

  def get_network_steering_angle(self, state_t):
    q_vector = self.online_Q.predict(state_t.reshape((1,)+IMG_SHAPE))
    print('Q values: {}'.format(q_vector))
    action_idx = np.argmax(q_vector)
    return (action_idx, self.action_space[action_idx])

  def remember(self, quadruple):
    """
    Add this state_t, (action_t_idx, action_t_angle), reward_t, state_t+1
    quadruple to replay memory.

    :param quadruple: state_t, action_t, reward_t, state_t+1 quadruple

    :returns: nada
    """
    self.replay_memory.remember(quadruple)

    
class ReplayMemory():
  """
  This class stores and handles all interactions with
  the quadruples of the form: (state_t,
                               (action_t_idx, action_t_steering),
                               reward_t,
                               state_t_plus_1)
  """
  def __init__(self, replay_memory_size=10**3, mini_batch_size=16):
    self.replay_memory = [ (np.empty(IMG_SHAPE, dtype=np.uint8),
                           (0, 0.0),
                           0.0,
                           np.empty(IMG_SHAPE, dtype=np.uint8)) ] * replay_memory_size
    self.replay_memory_size = replay_memory_size
    self.mini_batch_size = mini_batch_size
    self.idx_of_last_insert = -1
    self.num_memories = 0  # num memories over lifetime; not necess. num memories in mem 

  def remember(self, quadruple):
    self.idx_of_last_insert = (self.idx_of_last_insert + 1) % self.replay_memory_size
    self.replay_memory[self.idx_of_last_insert] = quadruple
    self.num_memories += 1

  def mini_batch_random_sample(self):
    """
    Randomly sample a batch quadruples from memory.
    """
    # get indices to sample
    if self.num_memories >= self.replay_memory_size:
      start_idx = random.randint(0, (self.replay_memory_size-1))
      end_idx = (start_idx + self.mini_batch_size - 1) % self.replay_memory_size
       # return those quadruples in that range of indices
      if start_idx > end_idx: # if sample wraps around from end to front of list
        return self.replay_memory[start_idx:] + self.replay_memory[0:(end_idx+1)]

      else: # can just sample normally
        return self.replay_memory[start_idx:(end_idx+1)]

      
    else:
      start_idx = random.randint(0, self.num_memories)
      end_idx = (start_idx + self.mini_batch_size - 1) % self.num_memories

      # return those quadruples in that range of indices
      if start_idx > end_idx: # if sample wraps around from end to front of list
        return self.replay_memory[start_idx:self.num_memories] + self.replay_memory[0:(end_idx+1)]

      else: # can just sample normally
        return self.replay_memory[start_idx:(end_idx+1)]


# thank you: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html
def init_car_controls():
  car_controls = airsim.CarControls()
  car_controls.throttle = 0.5125
  car_controls.steering = 0.0
  car_controls.is_manual_gear = True
  car_controls.manual_gear = 1  # should constrain speed

  return car_controls


print('Getting ready')
car_controls = init_car_controls()
replay_memory_size = 10000 # units=num images
reward_delay = 0.05 # seconds until know assign reward
episode_length =  int((1/reward_delay) * 180) # \neq to fps; fps depends on hardware
# right product should (roughly) be in seconds
mini_batch_train_size = 32

assert mini_batch_train_size <= replay_memory_size

C = 100 # copy weights to offline target Q net every n time steps
num_episodes = 100
epsilon = 1.0  # probability of selecting a random action/steering angle
episodic_epsilon_linear_decay_amount = (epsilon / num_episodes) # decay to 0
num_steering_angles = 10
train_frequency = 4 # train every n timesteps

print('On a perfect comptuer, the given settings indicate that each of the {} episodes '
      'should last about {} seconds ({} seconds in total)'.format(num_episodes,
                                                                  reward_delay*episode_length,
                                                                  reward_delay*episode_length*num_episodes))
                                                                  
                                                                  
                                                                  
      
driver_agent = DriverAgent(num_steering_angles=num_steering_angles,
                           replay_memory_size=replay_memory_size,
                           mini_batch_size=mini_batch_train_size,
                           gamma=0.98,
                           load_most_recent_weights=True)
airsim_env = AirSimEnv()
airsim_env.freeze_sim()  # freeze until enter loops


collisions_in_a_row = 0
collisions_in_a_row_before_restart = (1 / reward_delay) * 2 # seconds
epsilon += episodic_epsilon_linear_decay_amount # so that 1st episode uses epsilon
# for each episode
for episode_num in range(num_episodes):
  print('Starting episode {}'.format(episode_num+1))
  print('{} seconds (roughly) until all episodes finish'.format((num_episodes-episode_num)*episode_length*reward_delay))
  airsim_env.normal_reset()

  #  decay that epsilon value by const val
  
  epsilon -= episodic_epsilon_linear_decay_amount

  # for each time step
  for t in range(1, episode_length+1):
    print('\t time step {} of {}'.format(t, episode_length))
    airsim_env.unfreeze_sim()   # unfreeze for init loop or after previous step

    # Observation t
    car_state_t = airsim_env.get_car_state()

    # if car is starting to fall into oblivion or
    # was hit and is spinning out of control
    if car_state_t.kinematics_estimated.position.z_val > -0.5125 or \
       abs(car_state_t.kinematics_estimated.orientation.y_val) > 0.3125 or car_state_t.speed > 40.0: #kind-of works; 1.0 is upside down; m/s
      airsim_env.emergency_reset()

    state_t = airsim_env.get_environment_state() # panoramic image

    airsim_env.freeze_sim()  # freeze to select steering angle

    # Action t
    # if roll for random steering angle w/ probability epsilon
    if random.random() <= epsilon:
      action_t = driver_agent.get_random_steering_angle()
      print('\t Random Steering angle: {}'.format(action_t[1]))
    # else, consult the policy/Q network
    else:
      action_t = driver_agent.get_network_steering_angle(state_t)
      print('\t DQN Steering angle: {}'.format(action_t[1]))

    
    car_controls.steering = action_t[1] # 0 is idx, 1 is angle [-1, 1]
    
    airsim_env.unfreeze_sim()  # unfreeze to issue steering angle
    airsim_env.update_car_controls(car_controls)

    # Reward t
    time.sleep(reward_delay)  # wait for instructions to execute

    # State t+1
    car_state_t_plus_1 = airsim_env.get_car_state()

    # only record next state if this step is not last step in episode
    state_t_plus_1 = None
    if t != episode_length:
      state_t_plus_1 = airsim_env.get_environment_state()

    just_collided = airsim_env.get_collision_state()
    reward_t = driver_agent.current_reward(None, just_collided)

    airsim_env.freeze_sim()  # freeze to do training stuff

    driver_agent.remember( (state_t,
                            action_t,
                            reward_t,
                            state_t_plus_1) )

    # only train and/or copy weights after the 1st episode
    iteration_count = t + episode_length*episode_num
    if episode_num > 0 and (iteration_count % train_frequency) == 0: 
      driver_agent.lua_and_keras_train()

    if iteration_count % C == 0:
      driver_agent.copy_online_weights_to_offline()
      driver_agent.save_weights()

    if just_collided is True:
      collisions_in_a_row += 1
      if collisions_in_a_row >= collisions_in_a_row_before_restart:
        airsim_env.freeze_sim()
        airsim_env.normal_reset()
        airsim_env.unfreeze_sim()
        collisions_in_a_row = 0 # reset this counter
    else:
      collisions_in_a_row = 0
      

  # END inner for
# END OUTER for
