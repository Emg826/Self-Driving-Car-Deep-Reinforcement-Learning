"""
This file runs a client that get images from the simulation, reads them
into this program, and feeds them into a two linear regressors: one for
the throttle and one for the steering angle.

The purpose of this file is to learn how to get and use images from the
simulation and learn how to batch learn (which is necessary given that the
data is a continuous stream).

based on: https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md
"""

"""
ImageType values (first param in airsim.ImageRequest()). These are mostly
computer vision terms, so it helps to do a Google image search and see
what these images look like.

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

"""
Camera IDs

""  -- default camera
"0" -- front_center
"1" -- front_right
"2" -- front_left
"3" -- first person
"4" -- back_center
"""
import airsim
import numpy as np
import time

# how many past images to remember
NUM_IMAGES_TO_REMEMBER = 2e10  #2e10 means remember the past 51.2 seconds @ record interval=0.5 (20fps ?)
# number of images before retraining
NUM_IMAGES_BETWEEN_RETRAINS = 2e6   #2e6 means retrain every 12.8 seconds @ record interval=0.5 (20fps ?)


# client init
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# car controls struct init
car_controls = airsim.CarControls()

# collision info struct init
collision_info = client.simGetCollisionInfo()

# preallocate what memory will need to track images and targets
y = np.empty((NUM_IMAGES_TO_REMEMBER, 1))
while True:

    # PNG images of the scene -- type: binary string literal
    #sim_img_response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])

    # uncompressed RGBA array bytes (need to cast to use?)
    #sim_img_response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])

    # floating point uncompressed image -- type: list of float64s
    #sim_img_response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, True)])
    #sim_img = airsim.list_to_2d_float_array(sim_img_response.image_data_float,
    #                                        sim_img_response.width,
    #                                        sim_img_response.height)

    # uncompressed RGBA for numpy manipulation
    #sim_img_response = client.simGetImages([airsim.ImageRequest("0", AirSimImageType.Scene, False, False)])
    #sim_img_response = sim_img_response[0]
    #img1d = np.fromstring(sim_img_response.image_data_uint8,
    #                      dtype=np.uint8)

    sim_img_response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis),
                                            airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)])


        # here, I should probably error check to make sure that the simulation response is valid,
        # but I don't know what error to check for.



        car_state = client.getCarState()
	collision_info = client.simGetCollisionInfo()

	if collisions_in_a_row > 5:
		client.reset()  # restart car position

	# if collision, then try backing up for 2 seconds
	if collision_info.has_collided:
		car_controls = get_reverse_controls(car_controls)
		client.setCarControls(car_controls)
		time.sleep(2)
		collisions_in_a_row += 1
		continue

	collisions_in_a_row = 0  # won't get here unless didn't just collide
	# regardless of whether just backed up or not, go forward
	car_controls = get_forward_controls(car_controls, car_state)

	client.setCarControls(car_controls)

	# let the car drive with these given controls for a little while
	print('Speed {}. Gear: {}. Has Collided {}?'.format(car_state.speed,
														car_controls.manual_gear,
														collision_info.has_collided))
	time.sleep(3)
