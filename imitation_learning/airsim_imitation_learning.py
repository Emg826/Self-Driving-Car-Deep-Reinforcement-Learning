import time
from airsim_imitation_learning_data_collector import AirSimILDataCollectorAPI
import numpy as np

num_steering_angles = 7
time_between_steering_decisions = 0.25
safe_quit = True # save even if quit

api = AirSimILDataCollectorAPI(num_steering_angles=num_steering_angles)
arr_of_steering_angles = api.steering_angles

# key=keyboard key; val=steering angle idx [idx in arr_of_steering_angles]
key_bindings = {}
for number_key, steering_angle in zip(range(1, len(arr_of_steering_angles)+1), arr_of_steering_angles):
  key_bindings[number_key] = steering_angle  # not an index; is the actual angle in [-1, 1]

# make string once rather than each time the while loop iterates
save_message = 'save without quitting (note: clears data buffer)'
key_bindings['s'] = save_message
quit_message = 'quit with saving' if safe_quit else 'quit without saving'
key_bindings['q'] = quit_message
reset_message = 'reset car - back to starting point'
key_bindings['r'] = reset_message

key_binding_string = str(key_bindings)

# event loop
api.reset_vehicle()
user_input = None
rows_of_data = []
api.pause_sim()
while not api.arrived_at_destination():
  # ask for user input
  print('Please press one of these keys: (keyboard key : action)')
  print(key_binding_string)
  print()
  user_input = input()

  # process user input
  what_user_actually_wants = None
  try:
    what_user_actually_wants = key_bindings[user_input]
  except KeyError:  # except if invalid key
    continue

  # evaluate what user wants and then do it
  if what_user_actually_wants is quit_message:
    if safe_quit is True:
      np.save('npy/imitation_learning_rows_{}_.npy'.format(int(time.time())), np.array(rows_of_data))
    else:
      break

  elif what_user_actually_wants is save_message:
    np.save('npy/imitation_learning_rows_{}_.npy'.format(int(time.time())), np.array(rows_of_data))
    rows_of_data = []

  elif what_user_actually_wants is reset_message:
    api.reset_vehicle()
    time.sleep(1)  # avoid simulation crashes slightly more often?

  else:  # else will have selected a steering angle
    if str(int(user_input)) is user_input and \  # assert that user_input_is an int
      0 <= int(user_input)-1 < len(arr_of_steering_angles):  # valid steering angle
      api.set_steering_angle(int(user_input)-1)  # -1 since keyboard '1' is 0th idx in arr_of_steering_angles
    else:  # else, invalid steering idx
      continue

  # get sim data
  rows_of_data.append(api.get_sim_data())

  # execute
  api.unpause_sim()
  time.sleep(time_between_steering_decisions)
  api.pause()

print('The end. Thanks for playing!')
