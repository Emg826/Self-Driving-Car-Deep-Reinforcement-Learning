"""
Note: does not use linear regression.

This file runs a client that get images from the simulation, reads them
into this program, and feeds them through an incremental PCA and then
into a stochastic gradient descent classifier.

The purpose of this file is to learn how to get and use images from the
simulation and then batch learn on those images (which is necessary given that the
data is a continuous stream). This involves pausing before starting and after finishing
training.

based on: https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md
idea for collision-avoiding driver: https://github.com/simondlevy/AirSimTensorFlow
very helpful: https://github.com/Microsoft/AirSim/blob/master/PythonClient/airsim/types.py

ImageType values in airsim.ImageRequest(). These are mostly computer vision
terms, so it helps to Google image search and see what such images look like.

Scene = 0: raw image from camera?
DepthPlanner = 1: all pts in plan[e] parallel to camera have same depth
DepthPerspective = 2: depth from camera using projection ray that hits that pixel (point cloud)
DepthVis = 3: closest pixels --> black && >100m away pixels --> white pixels
DisparityNormalized = 4: ^ but normalized to [0, 1]?
Segmentation = 5: give specific meshes (road, lines, sidewalk, etc.) specific vals in image;
  allows for image to be "segmented" into meshes
SurfaceNormals = 6: ?
Infrared = 7: object ID 42 = (42,42,42); all else is grey scale
"""

import airsim
import numpy as np
import time
from sklearn.decomposition import IncrementalPCA # normalize then dimensionality reduction
from sklearn.linear_model import SGDClassifier # classify collision


# some important constants...
SEC_BETWEEN_RESETS = 50  # reset every n seconds for simplicity
SEC_BETWEEN_RETRAINS = 7 # retrain every n seconds
SEC_BETWEEN_IMG_REQUESTS = 0.2 # 0.05 means record 20 times per second
NUM_IMAGES_TO_REMEMBER = 75 # should be > SEC_BETWEEN_RETRAINS * (1 / SEC_BETWEEN_IMG_REQUESTS)
NUM_IMAGES_UNTIL_FEEDBACK = 3 # num images until get + or - feedback

PIXELS_IN_IMG = 36864   # number of pixels in the image

class CollisionAvoiderDriver():
  """
  Stochastic gradient descent classifier with a PCA preprocessor to ID
  imminent collisions. Both the PCA and the classifier iteratively learn,
  i.e., retraining does not result in the loss of the previous training
  session's params. The target of the classifier is
  collision=1/no collision=0.

  If imminent collision predicted, then break and swerve right. Otherwise,
  drive straight, full steam ahead.
  """
  def __init__(self):
    self.incremental_pca = IncrementalPCA(n_components=NUM_IMAGES_TO_REMEMBER)
    self.collision_classifier = SGDClassifier(max_iter=250,
                                              warm_start=True,
                                              alpha=0.5,
                                              loss='perceptron',
                                              shuffle=False)

  def train(self, X, y):
    """
    Partially fit the PCA and SGDClassifier.

    :param X: 2D array of images -- (num images, length of 1D image)
    :param y: array of collisions -- (num images, 1)
    """
    print("Training!")
    self.incremental_pca = self.incremental_pca.partial_fit(X)
    X_pca = self.incremental_pca.transform(X)

    self.collision_classifier = self.collision_classifier.partial_fit(X_pca,
                                                                      y,
                                                                      classes=[0, 1])  # collision=1

  def get_next_instructions(self, X, car_controls):
    """
    If collision is predicted, then

    :param X: list of len 1 with a 1D image as only entry
    :param car_controls: airsim.CarControls object from previous time step

    :returns: airsim.CarControls object with instructions for next time step
    """
    X_pca = self.incremental_pca.transform(X.reshape(1, -1))
    collision_imminent = self.collision_classifier.predict(X_pca)

    if collision_imminent:
      car_controls.throttle = -1.0
      car_controls.steering = 1.0
      car_controls.handbrake = True
    else:
      car_controls.throttle = 1.0
      car_controls.steering = 0.0
      car_controls.handbrake = False

    return car_controls

def fallen_into_oblivion(gt_kinematics):
  """
  Noticed a problem that the car will sometimes glitch through
  the ground of the map and cause the server to crash. To avoid
  crashses, try to reset once this happens.
  """
  # not sure why, but -z is up and +z is down?
  print('x={}, y={}, z={}'.format(gt_kinematics['position']['x_val'],
                                            gt_kinematics['position']['y_val'],
                                            gt_kinematics['position']['z_val']))

  # if fallen through the map floor; normal z coord is -0.65 to -0.8, so
  # -0.5 catches a fall into oblivion before it get out of hand.
  # now get over 700 rounds (probably more, but i didn't want to sit
  # around forever) of instruction giving; previously maybe half
  # that before oblivion fall (when did z > 0.1, which, then it was too late
  # to catch).
  if gt_kinematics['position']['z_val'] > -0.5:
    return True
  else:
    return False


# car controller init
driver = CollisionAvoiderDriver()

# client init
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# car controls struct init
car_controls = airsim.CarControls()

# collision info struct init
collision_info = client.simGetCollisionInfo()

# preallocate what memory will need to track images and targets
recent_images = np.zeros((NUM_IMAGES_TO_REMEMBER, PIXELS_IN_IMG),
                         dtype=np.float64)
most_recent_image_idx = NUM_IMAGES_UNTIL_FEEDBACK

recent_collisions = np.zeros((NUM_IMAGES_TO_REMEMBER),
                             dtype=np.uint8)
most_recent_collision_idx = 0

# track how far into the simulation
time_step = 0
last_reset_time = time.time()
last_train_time = time.time()

collisions_in_a_row = 0

while True:
  print('Round {}'.format(time_step+1))
  # request an image of the scene from the front facing camera
  sim_img_response = client.simGetImages([airsim.ImageRequest(camera_name='0',
                                                              image_type=airsim.ImageType.DepthPlanner,
                                                              pixels_as_float=True,
                                                              compress=False)])
  # extract 1D version of image from response obj
  sim_img = np.array(sim_img_response[0].image_data_float)
  recent_images[most_recent_image_idx] = sim_img
  most_recent_image_idx = (most_recent_image_idx + 1) % NUM_IMAGES_TO_REMEMBER

  # also want to get collided/not collided info for past instruction, so make a request
  collision_info = client.simGetCollisionInfo()
  recent_collisions[most_recent_collision_idx] = collision_info.has_collided
  most_recent_collision_idx = (most_recent_collision_idx + 1) % NUM_IMAGES_TO_REMEMBER


  # if there are too many collisions, the car can glitch through the
  # map and crash the Unreal engine; this should help avoid that
  # without affecting the training process too much
  if collision_info.has_collided:
    if collisions_in_a_row > 9:
      client.reset()
      collisions_in_a_row = 0
      last_reset_time = time.time()  # mark now as beginning of next episode
    else:
      collisions_in_a_row += 1
  else:
    collisions_in_a_row = 0


  # check if need to retrain or train for 1st time
  if (time.time() - last_train_time) > SEC_BETWEEN_RETRAINS or time_step == 0:
    client.simPause(True)
    driver.train(recent_images, recent_collisions)
    client.simPause(False)
    last_train_time = time.time()

  car_controls = driver.get_next_instructions(sim_img,
                                                            car_controls)

  client.setCarControls(car_controls)

  time_step += 1
  time.sleep(SEC_BETWEEN_IMG_REQUESTS) # let the car drive on these controls

  # if this epoch/episode is over
  if (time.time() - last_reset_time) > SEC_BETWEEN_RESETS:
    client.simPause(True)
    client.reset()  # restart car position
    client.simPause(False)
    last_reset_time = time.time()  # mark now as beginning of next episode

  # if car has fallen through the map
  if fallen_into_oblivion(client.simGetGroundTruthKinematics()):
    client.simPause(True)
    client.reset()
    client.simPause(False)
    last_reset_time = time.time()  # mark now as beginning of next episode

  # END while
