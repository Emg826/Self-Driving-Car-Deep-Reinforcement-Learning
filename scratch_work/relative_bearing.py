import math
import matplotlib.pyplot as plt
import numpy as np

def normalized_2_tuple(tup):
  magnitude_of_2_tup = math.sqrt(tup[0]**2 + tup[1]**2)
  if magnitude_of_2_tup != 0:
    # v / ||v||
    return  (tup[0] / magnitude_of_2_tup, tup[1] / magnitude_of_2_tup)
  else:
    return tup

def relative_bearing(car_yaw, car_position, destination_position):
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
  car_to_dest_vector = normalized_2_tuple(car_to_dest_vector)
  car_heading_vector = normalized_2_tuple(car_heading_vector)

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

  print(car_heading_vector, car_to_dest_vector)

  return relative_bearing

"""
# trivial cases
#straight ahead x direction  = 0
print(relative_bearing(0.0, (0.0, 0.0), (10.0, 0.0)))

#directly to right = + 1.57
print(relative_bearing(0.0, (0.0, 0.0), (0.0, 10.0)))

#directly to left  = -1.57
print(relative_bearing(0.0, (0.0, 0.0), (0.0, -10.0)))

#directly behind  = = ±3.14
print(relative_bearing(0.0, (0.0, 0.0), (-10.0, 0.0)))

# up and to right
print(relative_bearing(0.0, (0.0, 0.0), (10.0, 10.0)))


#45 degree angle, should get multiple of 45 degree (0.785 rads) angle for all these
print(relative_bearing(math.pi/4, (0.0, 0.0), (10.0, 0.0)))
print(relative_bearing(math.pi/4, (0.0, 0.0), (0.0, 10.0)))
print(relative_bearing(math.pi/4, (0.0, 0.0), (0.0, -10.0)))
print(relative_bearing(math.pi/4, (0.0, 0.0), (-10.0, 0)))

# all absolute values and signs of these tests are correct ! woohoo!
"""
