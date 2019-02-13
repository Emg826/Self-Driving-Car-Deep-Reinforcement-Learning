"""
So, the data from imitation_learning has x and y together, has
done no preprocessing to x, and needs to still convert scenes
RGB to grayscale; do some of that here.
"""


import numpy as np
import cv2
import time

filepath_to_npy_data = 'npy/imitation_learning_rows_1549589372_.npy'
all_rows = np.load(filepath_to_npy_data)

x = all_rows[:,0:-1]
y = all_rows[:, -1]

print('x shape', x.shape)
print('y shape', y.shape)

rgb_to_grayscale = lambda scene_img: cv2.cvtColor(scene_img, cv2.COLOR_RGB2GRAY)

NUM_STEERING_ANGLES = 7
multi_output_y = np.zeros((y.shape[0], NUM_STEERING_ANGLES), dtype=np.int)
scenes = []  # keras multiinput nmodel expects [all_scenes, all_depths, all_miscs]
depths = []
miscs = []
for row_idx in range(0, x.shape[0]):
  # scene gray scale and add channel dimension
  scene = rgb_to_grayscale(x[row_idx, 0])
  scenes.append(scene.reshape(scene.shape[0], scene.shape[1], 1))

  # depth  add channel dimension
  depths.append(x[row_idx, 1].reshape(x[row_idx, 1].shape[0], x[row_idx, 1].shape[1], 1))

  # misc data add channel
  miscs.append(x[row_idx, 2].reshape(x[row_idx, 2].shape[0], 1))

  multi_output_y[row_idx, y[row_idx]] = 1

  
print('scene image shape after rgb2gray', scenes[0].shape)


time_stamp = int(time.time())
np.save('npy/x_scenes_{}.npy'.format(time_stamp), np.array(scenes))
np.save('npy/x_depths_{}.npy'.format(time_stamp), np.array(depths))
np.save('npy/x_miscs_{}.npy'.format(time_stamp),  np.array(miscs))

np.save('npy/y_{}.npy'.format(time_stamp), np.array(multi_output_y))

