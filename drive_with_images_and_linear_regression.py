"""
This file runs a client that get images from the simulation, reads them
into this program, and feeds them into a two linear regressors: one for
the throttle and one for the steering angle.

The purpose of this file is to learn how to get and use images from the
simulation and learn how to batch learn (which is necessary given that the
data is a continuous stream).

based on: https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md
idea for driver based on: https://github.com/simondlevy/AirSimTensorFlow
very helpful: https://github.com/Microsoft/AirSim/blob/master/PythonClient/airsim/types.py
"""
"""
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

SEC_BETWEEN_RESETS = 30  # reset every n seconds for simplicity
SEC_BETWEEN_RECORDS = 0.05 # 0.05 means record 20 times per second
NUM_IMAGES_TO_REMEMBER = 100  #2e10 means remember the past 51.2 seconds @ record interval=0.5 (20fps ?) number of images before retraining
NUM_IMAGES_BETWEEN_RETRAINS = 40   # 2e6 means retrain every 12.8 seconds @ record interval=0.5 (20fps ?)
NUM_IMAGES_UNTIL_FEEDBACK = 10 # num images until get + or - feedback

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
        self.incremental_pca = IncrementalPCA()
        self.collision_classifier = SGDClassifier(max_iter=500, wamr_start=True)

    def train(self, X, y):
        """
        Partially fit the PCA and SGDClassifier.

        :param X: 2D array of images -- (num images, length of 1D image)
        :param y: array of collisions -- (num images, 1)
        """
        self.incremental_pca.partial_fit(X)
        X_pca = self.incremental_pca.transform(X)
        self.collision_classifier.partial_fit(X_pca, y)

    def get_next_instructions(self, X, car_controls):
        """
        If collision is predicted, then

        :param X: list of len 1 with a 1D image as only entry
        :param car_controls: airsim.CarControls object from previous time step

        :returns: airsim.CarControls object with instructions for next time step
        """
        X_pca = self.incremental_pca.transform(X)
        collision_imminent = self.collision_classifier.predict(X_pca)

        if collision_imminent:
            car_controls.throttle = -1.0
            car_controls.steering = 1.0
            car_controls.handbrake = True
        else:
            car_controls.throttle = 0.5
            car_controls.steering = 0.0
            car_controls.handbrake = False

        return car_controls


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
recent_images = np.empty((NUM_IMAGES_TO_REMEMBER, 1))
most_recent_image_idx = NUM_IMAGES_UNTIL_FEEDBACK

recent_collisions = np.empty((NUM_IMAGES_TO_REMEMBER, 1))
most_recent_collision_idx = 0

# track how far into the simulation
time_step = 0
last_reset_time = time.time()
num_times_trained = 0

while True:
    # read in an image of the scene from the front facing camera
    sim_img_response = client.simGetImages([airsim.ImageRequest(camera_name='0',
                                                                image_type=airsim.ImageType.Scene,
                                                                pixels_as_float=True,
                                                                compress=False)])
    recent_images[most_recent_image_idx] = sim_img_response.image_data_float
    most_recent_image_idx = (most_recent_image_idx + 1) % NUM_IMAGES_TO_REMEMBER

    # also want to get collided/not collided info for past instruction
    collision_info = client.simGetCollisionInfo()
    recent_collsions[most_recent_collision_idx] = collision_info.has_collided
    most_recent_collision_idx = (most_recent_collision_idx + 1) % NUM_IMAGES_TO_REMEMBER

    # check if need to retrain
    if time_step % NUM_IMAGES_BETWEEN_RETRAINS == 0 or time_step == 0:
            driver.train(recent_images, recent_collisions)
            num_times_trained += 1

    # only get next controls from driver obj if trained at least 1 time
    if num_times_trained > 0:
        car_controls = driver.get_next_instructions(sim_img_response.image_data_float,
                                                    car_controls)

    client.setCarControls(car_controls)

    time_step += 1
    time.sleep(SEC_BETWEEN_RECORDS) # let the car drive on these controls

    # if this epoch/episode is over
    if time.time() - last_reset_time > SEC_BETWEEN_RESETS:
		client.reset()  # restart car position
        time.sleep(1)  # wait for reset to complete
        last_reset_time = time.time()  # mark now as beginning of next episode

    # END while
