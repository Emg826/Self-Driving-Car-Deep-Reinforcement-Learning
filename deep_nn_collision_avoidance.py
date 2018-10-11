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

Note*: this DQN agent does not stack frames like in the DeepMind paper


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

from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, MaxPooling2D
#https://web.stanford.edu/class/cs20si/2017/lectures/slides_14.pdf

IMG_SHAPE = (260, 1190,  4) # H x W x NUM_CHANNELS
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
    self.client.confirmConnection()
    self.client.enableApiControl(True)

    self.left_cam_name = '2'
    self.right_cam_name = '1'
    self.forward_cam_name = '0'
    self.backward_cam_name = '4'

    self.setup_my_cameras()
    self.emergency_reset_coords = airsim.Vector3r(x_val=0,
                                                  y_val=0,
                                                  z_val=-0.7) # +x is fwd, +y is right, -z is up

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
    self.client.reset()
    self.client.simPause(False)

  def freeze_sim(self):
    self.client.simPause(True)

  def unfreeze_sim(self):
    self.client.simPause(False)



class DriverAgent():
  """
  The driver of the vehicle, a deep Q network that uses
  deep Q learning.
  """
  def __init__(self, num_steering_angles, replay_memory_size,
               mini_batch_size=32, gamma=0.98, n_points_replay=3):
    """
    Initialize the Deep Q driving agent
    
    :param num_steering_angles: number of steering angles, equidistant from each other, in [-1, 1]
    :param replay_memory_size: number of previous states to remember, sample from this many states
    :param optimizer: a keras.optimizer object w/ the class func get_updates()
    :param mini_batch_size: size of a batch to randomly sample from replaymemory
    :param gamma: discount rate of next reward (and determines all future rewards)
    """
    # prealloc replay mem of 4-tuples of the form: (state_t, action_t, reward_t, state_t+1)
    self.replay_memory = [ (np.empty(IMG_SHAPE, dtype=np.uint8),
                            0.0,
                            0.0,
                            np.empty(IMG_SHAPE, dtype=np.uint8)) ] * replay_memory_size

    self.replay_memory_size = replay_memory_size
    self.mini_batch_size = mini_batch_size
    self.gamma = gamma
    self.next_replay_memory_insert_idx = 0
    
    # airsim steering angles are in the interval [-1, 1] for some reason
    self.num_steering_angles = num_steering_angles
    self.action_space = np.arange(-1.0, 1.001, 2.0/num_steering_angles).tolist()
    # ^ is a list for because actions stored are steering angles, not ouput node idx
    # and lists have method .index(val_in_list) [used in training]

    # neural network to approximate the policy/Q function
    self.online_Q = Sequential()
    self.online_Q.add(Convolution2D(64, kernel_size=8, strides=8, input_shape=IMG_SHAPE,
                                 data_format='channels_last', activation='relu'))
    self.online_Q.add(MaxPooling2D(pool_size=4, strides=4))
    self.online_Q.add(Flatten())
    self.online_Q.add(Dense(256, activation='relu'))
    self.online_Q.add(Dense(num_steering_angles))  # 1 output per steering angle

    self.offline_Q = self.online_Q # target network -- used to update weights
    self.offline_Q.compile(optimizer='rmsprop', loss='mse')
    self.online_Q.compile(optimizer='rmsprop', loss='mse') # used for selecting actions -- copies weights from offline
    # ^ optimizer and loss don't really matter since will be manually updating weights (not use model.fit())

  def mini_batch_sample(self):
    """
    Get a random chunk of self.mini_batch_size 4-tuples from replay memory,
    each of the form (state_t, action_t, reward_t, state_t+1).
    """
    # sampling doesn't "wrap around" (e.g., idx's 31, 0, 1 ...); can do later
    idx_before_could_seg_fault = self.replay_memory_size - self.mini_batch_size
    start_idx = random.randint(0, idx_before_could_seg_fault)
    end_idx = start_idx + (self.mini_batch_size-1)

    return self.replay_memory[start_idx:(end_idx+1)]

  def current_reward(self, car_state):
    """
    Custom reward function

    :param car_state: airsim.CarState() of car in sim
    
    :returns: the current reward given the state of the car

    *Note: want to keep the rewards in [-1, 1] (see DeepMind Nature paper)
    """
    if car_state.collision.has_collided is True:
      # collisions attrib has object_id (collided w/), so
      # very easily could modify and see if collided with person
      # or vehicle or building or curb or etc.
      return -1.0
    else:
      return 0.5

  def train(self):
    """
    Train on a minibatch of quadruples from replay memory.
    """
    # step 1. get a random minibatch from r memory
    mini_batch = self.mini_batch_sample()

    # step 2. get targets from offline and online Q predictions
    # important note: term is just a 1 or a 0 (see lua code on deepmind github),
    # which is just more efficient at calc target (no if-else) [i don't use term here]
    targets = []
    predictions = []
    q_j_plus_1_maxes = []
    for quadruple in mini_batch:
      
      s_j, a_j, r_j, s_j_plus_1 = quadruple

      # if s_t is the last state in the episode
      if s_j_plus_1 is None:  # (1 - terminal=1) = 0
        targets.append(r_j)
      # else, get a target Q from the offline network (target network)
      else: # (1 - terminal=0) = 1
        # parallel on github: delta = r:clone():float() and delta:add(q2)
        q_j_plus_1_max = np.max(self.offline_Q.predict(s_j_plus_1))
        q_j_plus_1_maxes.append(q_j_plus_1_max)
        targets.append(r_j + self.gamma * q_j_plus_1_max)

      # parallel on github: delta:add(-1, q)
      # for whatever reason
      predictions.append(self.online_Q.predict(s_j)[self.action_space.index(a_j)])

    # note: did not rescale rewards in min,max range
    # step 3. gradient descent on error^2 w/ rspct to online weights
    delta = [target - prediction for target, prediction in zip(targets, predictions)]
    return None # don't use this func


  def my_train_with_keras(self):
    """
    This is my attempt to combine python's Keras and the lua code
    from DeepMind's github that contains the deep q network code.
    The hard part of that code is manually applying the gradients,
    but I think Keras should be able to handle that.
    """
    mini_batch = self.mini_batch_sample()

    # this format is standard and expected throughout this program
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

    self.online_Q.fit(np.asarray(s), targets, shuffle=False, verbose=0)


  def copy_online_weights_to_offline(self):
    """
    Copy online network's weights to offline network. Occurs every
    C time steps (C is not an attribute of this class [though it could be later])
    """
    self.offline_Q.set_weights(self.online_Q.get_weights())

  def save_weights(self, save_offline_weigths=False):
    """
    Save the model weights to h5 file that can be loaded in to a model
    with the exact same architecture model should the program crash
    """
    directory = '/model_weights'
    if os.path.isdir(directory) is False:
      os.mkdir(directory)

    time_stamp = int(time.time())
    if save_offline_weights:
      offline_weight_filename = 'offline_{}.h5'.format(time_stamp)
      self.offline_Q.save_weights(os.path.join(directory, offline_weight_filename))
      print('Offline weights saved at {}'.format(time_stamp))

    else:
      online_weight_filename = 'online_{}.h5'.format(time_stamp)
      self.online_Q.save_weights(os.path.join(directory, online_weight_filename))
      print('Online weights saved at {}'.format(time_stamp))    

  def get_random_steering_angle(self):
    action_idx = random.randint(0, self.num_steering_angles-1)
    return (action_idx, self.action_space[action_idx])

  def get_steering_angle(self, state_t):
    action_idx = np.argmax(self.online_Q.predict(state_t.reshape((1,)+IMG_SHAPE)))
    return (action_idx, self.action_space[action_idx])

  def remember(self, quadruple):
    """
    Add this state_t, action_t, reward_t, state_t+1 quadruple to
    replay memory.

    :param quadruple: state_t, action_t, reward_t, state_t+1 quadruple

    :returns: nada
    """
    self.replay_memory[self.next_replay_memory_insert_idx] = quadruple
    self.next_replay_memory_insert_idx = (self.next_replay_memory_insert_idx+1)\
                                         % self.replay_memory_size


# thank you: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html
def init_car_controls():
  car_controls = airsim.CarControls()
  car_controls.throttle = 0.5
  car_controls.steering = 0.0
  car_controls.is_manual_gear = True
  car_controls.manual_gear = 1  # should constrain speed

  return car_controls



print('Getting ready')
car_controls = init_car_controls()
replay_memory_size = 60 # units=num images
episode_length = 80 # no idea what this means for fps
assert episode_length > replay_memory_size  # for simplicity's sake; don't actually want this
mini_batch_train_size = 4
assert mini_batch_train_size <= replay_memory_size
C = 30 # copy weights to offline target Q net every n time steps
num_episodes = 100
epsilon = 1.0  # probability of selecting a random action/steering angle
episodic_epsilon_linear_decay_amount = (epsilon / num_episodes) # decay to 0
num_steering_angles = 10
reward_delay = 0.05 # seconds until know assign reward


driver_agent = DriverAgent(num_steering_angles=num_steering_angles,
                           replay_memory_size=replay_memory_size,
                           mini_batch_size=mini_batch_train_size,
                           gamma=0.98)
airsim_env = AirSimEnv()
airsim_env.freeze_sim()  # freeze until enter loops


# for each episode (arbitrary length of time)
for episode_num in range(num_episodes):
  print('Starting episode {}'.format(episode_num+1))
  airsim_env.normal_reset()

  #  decay that epsilon value by const val
  epsilon -= episodic_epsilon_linear_decay_amount

  # for each time step
  for t in range(1, episode_length+1):
    print('\t time step {}'.format(t))
    airsim_env.unfreeze_sim()   # unfreeze for init loop or after previous step

    # Observation t
    car_state_t = airsim_env.get_car_state()

    # if car is starting to fall into oblivion
    if car_state_t.kinematics_estimated.position.z_val > -0.4:
      airsim_env.emergency_reset()

    state_t = airsim_env.get_environment_state() # panoramic image

    airsim_env.freeze_sim()  # freeze to select steering angle

    # Action t
    # if roll for random steering angle w/ probability epsilon
    if random.random() < epsilon:
      action_t = driver_agent.get_random_steering_angle()
      print('\t Random Steering angle: {}'.format(action_t[1]))
    # else, consult the policy/Q network
    else:
      action_t = driver_agent.get_steering_angle(state_t)
      print('\t DQN Steering angle: {}'.format(action_t[1]))

    
    car_controls.steering = action_t[1] # 0 is idx, 1 is angle [-1, 1]
    
    airsim_env.unfreeze_sim()  # unfreeze to issue steering angle
    airsim_env.update_car_controls(car_controls)

    # Reward t
    # time.sleep(reward_delay)  # wait for instructions to execute

    # State t+1
    car_state_t_plus_1 = airsim_env.get_car_state()

    # only record next state if this step is not last step in episode
    state_t_plus_1 = None
    if t != episode_length:
      state_t_plus_1 = airsim_env.get_environment_state()

    airsim_env.freeze_sim()  # freeze to do training stuff

    reward_t = driver_agent.current_reward(car_state_t_plus_1)
    driver_agent.remember( (state_t,
                            action_t,
                            reward_t,
                            state_t_plus_1) )

    # only train and/or copy weights after the 1st episode
    iteration_count = t + episode_length*episode_num
    if episode_num > 0: 
      driver_agent.my_train_with_keras()

      if iteration_count % C == 0:
        driver_agent.copy_online_weights_to_offline()

  # END inner for
# END OUTER for
