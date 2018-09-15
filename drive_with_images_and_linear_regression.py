"""
This file runs a client that get images from the simulation, reads them
into this program, and feeds them into a two linear regressors: one for
the throttle and one for the steering angle.

The purpose of this file is to learn how to get and use images from the 
simulation and learn how to batch learn (which is necessary given that the 
data is a continuous stream). 
"""

import airsim
import time

# how many past images to remember
NUM_IMAGES_TO_REMEMBER = 2e12  #2e12 means remember the past 204.8 seconds @ record interval=0.5 (20fps ?)
RETRAIN_EVERY_N_IMAGES = 2e6   #2e6 means retrain every 12.8 seconds @ record interval=0.5 (20fps ?)



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






