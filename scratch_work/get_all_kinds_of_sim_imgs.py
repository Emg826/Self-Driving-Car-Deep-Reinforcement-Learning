"""
This is more or less just a file for quickly getting simulation images.
As such, it's patchwork, commented-out code from over the course of this project.


Output an image of each type from simulator just so to see what it looks like.

ImageType values in airsim.ImageRequest(). These are mostly computer vision
terms, so it helps to Google image search and see what such images look like.

Scene = 0: raw image from camera?
DepthPlanner = 1: all pts in plan[e] parallel to camera have same depth - pixels are in meters, i.e.,
  they're not colors or shades of gray
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
import cv2
import time
import math

# client init
client = airsim.CarClient()
client.confirmConnection()


left_cam_name = '2'
right_cam_name = '1'
forward_cam_name = '0'
backward_cam_name = '4'


# same cam orientations from airsim_env.py
"""
client.simSetCameraOrientation(left_cam_name,
                                        airsim.Vector3r(0.0, 0.0, -0.68))
client.simSetCameraOrientation(right_cam_name,
                                   airsim.Vector3r(0.0, 0.0, 0.68))
client.simSetCameraOrientation(forward_cam_name,
                                        airsim.Vector3r(0.0, 0.0, 0.0))
client.simSetCameraOrientation(backward_cam_name,
                                        airsim.Vector3r(0.0, 0.0, 11.5))
"""

# in radians, left is [-3.14, 0] and right is [0, 3.14] and 0 is fwd
# 10deg = 0.17radians, 15deg = 0.26radians, 30deg = 0.52radians, 45deg = 0.78radians (appx)

# now want a 256x384 image hxw
# 3 images, only going to have (1-0.325-0.375)=0.3 proportion of image
# so, width of image reduced from 384 to 384/3 = 128 for the width of camera
# height = 256 * (1/(1-0.325-0.375)) = 853 height


left_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=-0.775, roll=0.0)
forward_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=0.0, roll=0.0)
right_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=0.775, roll=0.0)
backward_cam_orientation = airsim.to_quaternion(pitch=-0.17, yaw=0.0, roll=0.0)



client.simSetCameraOrientation(left_cam_name, left_cam_orientation)
client.simSetCameraOrientation(right_cam_name, right_cam_orientation)
client.simSetCameraOrientation(forward_cam_name, forward_cam_orientation)
#client.simSetCameraOrientation(backward_cam_name, airsim.Vector3r(0.0, 0.0, 11.5))

"""
sim_img_responses = client.simGetImages([airsim.ImageRequest(camera_name=left_cam_name,
                                                                                      image_type=airsim.ImageType.Scene,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name=forward_cam_name,
                                                                                      image_type=airsim.ImageType.Scene,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name=right_cam_name,
                                                                                      image_type=airsim.ImageType.Scene,
                                                                                      pixels_as_float=False,
                                                                                      compress=False)])

img = np.concatenate([np.fromstring(img_response_obj.image_data_uint8,
                                    dtype=np.uint8).reshape(img_response_obj.height,
                                                            img_response_obj.width,
                                                            4) for img_response_obj in sim_img_responses],
                     axis=1)

fraction_of_top_of_scene_to_drop = 0.05
fraction_of_bottom_of_scene_to_drop = 0.1
first_scene_row_idx = int(img.shape[0] * fraction_of_top_of_scene_to_drop)
last_scene_row_idx = int(img.shape[0] * (1-fraction_of_bottom_of_scene_to_drop))
img = img[first_scene_row_idx: last_scene_row_idx]

img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)

cv2.imwrite('composite.jpg', img)
"""

sim_img_responses = client.simGetImages([airsim.ImageRequest(camera_name=left_cam_name,
                                                                                      image_type=airsim.ImageType.DepthPlanner,
                                                                                      pixels_as_float=True,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name=forward_cam_name,
                                                                                      image_type=airsim.ImageType.DepthPlanner,
                                                                                      pixels_as_float=True,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name=right_cam_name,
                                                                                      image_type=airsim.ImageType.DepthPlanner,
                                                                                      pixels_as_float=True,
                                                                                      compress=False)])
print(len(sim_img_responses))
img = np.concatenate([airsim.list_to_2d_float_array(img_response_obj.image_data_float,
                                                    img_response_obj.width,
                                                    img_response_obj.height,) for img_response_obj in sim_img_responses],
                     axis=1)

fraction_of_top_of_scene_to_drop = 0.00
fraction_of_bottom_of_scene_to_drop = 0.4
first_scene_row_idx = int(img.shape[0] * fraction_of_top_of_scene_to_drop)
last_scene_row_idx = int(img.shape[0] * (1-fraction_of_bottom_of_scene_to_drop))
img = img[first_scene_row_idx: last_scene_row_idx]


# 255 * 2 = 510
PHI = lambda pixel: 255.0 if pixel < 3.90 else 450.0 / pixel
# this function is about 40% faster than the previous min/max func (below)
#PHI = lambda pixel:   min(1024.0 / (pixel+3.0), 255.0)

# https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
# WOW!!!! vectorizing PHI makes it 10x faster than any of the other implementations!
ct = 0
PHI_VECTORIZED = np.vectorize(PHI) #this would be done upon init of env, so not include in timer
start = time.time()

while ct < 10:
  img_ = img.copy()
  img_ = PHI_VECTORIZED(img_)
  ct += 1

print('time to apply phi 10 times via np vectorize:', time.time() - start)
cv2.imwrite('depth_planner_preprocessed_vectorized.jpg', img_)




"""

ct = 0
start = time.time()
while ct < 10:
  img_ = img.copy()
  for row_idx in range(0, img_.shape[0]):
    this_row_but_preprocessed = []
    for col_idx in  range(0, img_.shape[1]):
      this_row_but_preprocessed.append(PHI(img_[row_idx, col_idx]))

    img_[row_idx] = this_row_but_preprocessed
  ct += 1
print('time to apply phi 10 times via for loops but row by row:', time.time() - start)



ct = 0
start = time.time()
while ct < 10:
  img_ = img.copy()
  for row_idx in range(0, img_.shape[0]):
    img_[row_idx] = [PHI(pixel_value) for pixel_value in img_[row_idx]]

  ct += 1
print('time to apply phi 10 times via for loops but row by row list comp:', time.time() - start)


ct = 0
start = time.time()
while ct < 10:
  img_ = img.copy()
  for row_idx in range(0, img_.shape[0]):
    for col_idx in  range(0, img_.shape[1]):
      img_[row_idx, col_idx] = PHI(img_[row_idx, col_idx])
  ct += 1
print('time to apply phi 10 times via for loops:', time.time() - start)



# https://docs.scipy.org/doc/numpy/reference/generated/numpy.apply_along_axis.html
ct = 0
start = time.time()
PHI_FOR_ROWS = lambda row: [min(1024.0 / (pixel_value+3.0), 255.0) for pixel_value in row]
while ct < 10:
  img_ = img.copy()
  img_ = np.apply_along_axis(PHI_FOR_ROWS, 0, img_)
  ct += 1
print('time to apply phi 10 times via np apply_along_axis:', time.time() - start)


cv2.imwrite('composite.jpg', img)

"""











# what i actually went with
sim_img_response = client.simGetImages([airsim.ImageRequest(camera_name='0', image_type=airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False)])
img = airsim.list_to_2d_float_array(sim_img_response[0].image_data_float, sim_img_response[0].width, sim_img_response[0].height)
#PHI = lambda pixel: min(255.0, max(0, 75.0*math.log(pixel-0.5, math.e)))

# 40 ft (which is <= cond) is cutoff of sensor; @ 12mph (17.6 ft/s),
# would take about 2 seconds to reach end of current img
#295 = 255 + 40 && sqrt(40) = 6.32
"""
PHI = lambda pixel: min(255.0, ( 295.0 / (math.sqrt(max(0.001, pixel -0.475)))) - 6.32) if pixel <= 40.0 else 0.0

# apply PHI to each pixel
for row_idx in range(0, img.shape[0]):
  for col_idx in range(0, img.shape[1]):
    img[row_idx][col_idx] = PHI(img[row_idx][col_idx])

print(img[0:15])
print(img[int(img.shape[0]*.7):int(img.shape[0]*.8)])
        
#cv2.imwrite('depthPlannerMin255orMax0or80timeslnx-pt5.jpg', img)
cv2.imwrite('depthPlannerMin0orMax0or70timeslnabsxminuspt7.jpg', img)
cv2.imwrite('minus_top_third_and_minus_bottom_tenth.jpg', img[int(img.shape[0]*0.20) : int(img.shape[0]*0.8)] )
"""
"""
sim_img_responses = client.simGetImages([airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Scene,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.DepthPlanner,
                                                                                      pixels_as_float=True,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Segmentation,
                                                                                      pixels_as_float=False,
                                                                                      compress=False)])

# apply to raw imgs
alter_pixel = lambda pixel:  min(1024.0 / (pixel+3.0), 255.0)
fraction_of_top_of_scene_to_drop=0.45
fraction_of_bottom_of_scene_to_drop=0.1
fraction_of_top_of_depth_to_drop=0.375
fraction_of_bottom_of_depth_to_drop=0.375

# only using middle 25% of depth image  - so make square by width / 0.25 ==> 4x width?
# and only using middle 45% of scene image - so make squar by width / 045 ==> 2.2x width?
# so, let's try: width of scene = 512, so then height = 1137
#              ; : width of depth = 384, so then height = 384 / 0.25 = 1536

img_names = ['Scene', 'DepthPlanner', 'Segmentation']
for img_response_obj, img_name in zip(sim_img_responses, img_names):
  height = img_response_obj.height
  width = img_response_obj.width

  img = None
  if img_name == 'DepthPlanner':
    img = airsim.list_to_2d_float_array(img_response_obj.image_data_float, width, height)

    first_depth_planner_row_idx = int(height * fraction_of_top_of_depth_to_drop)
    last_depth_planner_row_idx = int(height * (1-fraction_of_bottom_of_depth_to_drop))
    img = img[ first_depth_planner_row_idx : last_depth_planner_row_idx]

    for row_idx in range(0, img.shape[0]):
      for col_idx in  range(0, img.shape[1]):
        img[row_idx][col_idx] = alter_pixel(img[row_idx][col_idx])


  elif img_name == 'Segmentation' or img_name == 'Scene':
    img = np.fromstring(img_response_obj.image_data_uint8, dtype=np.uint8).reshape(height, width, 4)
    first_scene_row_idx = int(height * fraction_of_top_of_scene_to_drop)
    last_scene_row_idx = int(height * (1-fraction_of_bottom_of_scene_to_drop))

    img = img[first_scene_row_idx: last_scene_row_idx]

    
  cv2.imwrite('{}.jpg'.format(img_name), img)
"""
