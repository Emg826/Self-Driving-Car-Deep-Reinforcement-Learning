"""
This program instantiates a client through which this program can communicate 
with the server. All that this program does is tell the car to 
go forward in a random direction at about 5 mph, go in reverse if there 
was a collision, or reset the car if the car gets stuck.

The purpose of this file is to become familiar with connecting a client
to an extant server, as well as to understand how to use the car state and 
car controls for the simulation. As such, there is no image gathering and
no machine learning used here.

Based on: https://github.com/Microsoft/AirSim/blob/master/docs/apis.md

09/14/2018 - EG
"""


import airsim
import time
import numpy as np


def get_reverse_controls(car_controls):
	"""
	Set the car controls to reverse
	:param car_state: structure that comes from airsim.CarControls()
	
	:returns: car control struct w/ values to reverse the car
	"""
	car_controls.throttle = -0.3
	car_controls.is_manual_gear = True
	car_controls.manual_gear = -1  # this means reverse to the API
	car_controls.steering = 0 # back up straight

	return car_controls


def get_forward_controls(car_controls, car_state):
	"""
	Get controls to put the car into a forward motion. The steering 
	direction is random.

	:param car_controls: structure that comes from airsim.CarControls()

	:returns: car control struct w/ values to drive the car forward
	"""
	if car_state.speed < 5.0:
		car_controls.throttle = 0.5
	else:
		car_controls.throttle = 0

	car_controls.is_manual_gear = False
	car_controls.steering = np.random.normal(loc=0.0, scale=0.2)
	
	return car_controls




# this py program is where the client resides, essentially, so instantiate a client
client = airsim.CarClient()

# try to connect to the server (which should already be running)
client.confirmConnection()

# not sure if ^ will wait for a connection even if connection fails, but 
# i guess that this next function will connect to tell the server 
# that this program wants to communicate through the API
client.enableApiControl(True)

# this comes from the client, it's just a structure w/ 
# attributes: brake (true or false), throttle (true or false),
# and steering angle
car_controls = airsim.CarControls()

# IRL, can't really get this info, but for the purposes of this 
# basic interaction, getting this information is OK.
# see car_collision.py for more (in ..\PythonClient\car)
# NOTE: functions that begin w/ 'sim' are not available IRL 
collision_info = client.simGetCollisionInfo() 
# collision_info is also a structure w/ attributes: position, normal, 
# impact_point, penetration_depth, object_name, object_id, has_collieded


# run until this program is canceled 
collisions_in_a_row = 0
while True: 

    # based on: https://github.com/Microsoft/AirSim/blob/master/docs/apis.md
	# get the state of the car: current speed, current gear, and the 6 kinematic
	# quants: position, orientation, linear velocity, angular velocity,
	# linear acceleration, angular acceleration
	car_state = client.getCarState()  # the program asks the client object to
	# ask the server to get the car state? 


	# want to know if collided or not so know whether to reverse or not
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






