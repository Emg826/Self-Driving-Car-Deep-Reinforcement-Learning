"""
The deep_nn_weights folder can get really large if the simulation is left
to run for a long time and/or if the weights are saved frequently. Therefore,
run this file with command line argument_1 = folder name to delete the oldest
weight file when argument_2=max-num-new-weights is reached or exceeded.

Note: only tracks weights added since this program started running; does not
delete weights if the weights were already there before running this program.

Example Run: python weight_control.py dir_from_wd num_new_weights_to_keep 
"""

import os
import time
import sys

args = sys.argv # list w/ first elem name of program; rest are args 

if len(args) != 3:
  print('python  weight_patrol dir_from_wd  num_new_weights_to_keep')
  sys.exit(1)

full_weights_dir = os.path.join(os.getcwd(), args[1])  # abs filepath
if os.path.exists(full_weights_dir) is False:
  print('{} is not a path'.format(full_weights_dir))
  sys.exit(1)

num_new_weights_to_keep = int(args[2])
if num_new_weights_to_keep < 1:
  print('num new weights to keep must be greater than or equal to 1')
  sys.exit(1)

list_initial_files_in_weights_dir = [os.path.join(full_weights_dir, f) for f in os.listdir(full_weights_dir)]
while True:
  time.sleep(30)

  list_current_files_in_weights_dir = [os.path.join(full_weights_dir, f) for f in os.listdir(full_weights_dir)]

  # get list of new files
  new_weight_files = []
  for file_path in list_current_files_in_weights_dir:
    if file_path not in list_initial_files_in_weights_dir and '.h5' in file_path:
      new_weight_files.append(file_path)

  # check if reached new file capacity
  # if too many new weights, then delete oldest
  if  len(new_weight_files) > num_new_weights_to_keep:
    # determine oldest weights file and delete it
    oldest_file_path = None
    oldest_file_path_ctime = sys.maxsize
    for this_file_path in new_weight_files:
      this_file_path_ctime = os.stat(this_file_path).st_ctime

      if this_file_path_ctime < oldest_file_path_ctime:
        oldest_file_path = this_file_path
        oldest_file_path_ctime = this_file_path_ctime

    # delete the oldest file
    print('deleting {}'.format(oldest_file_path))
    os.remove(os.path.join(full_weights_dir, oldest_file_path))

  # else continue
  else:
    continue

  

  
