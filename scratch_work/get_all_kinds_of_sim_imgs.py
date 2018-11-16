"""
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
client.enableApiControl(True)


# what i actually went with
sim_img_response = client.simGetImages([airsim.ImageRequest(camera_name='0', image_type=airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False)])
img = airsim.list_to_2d_float_array(sim_img_response[0].image_data_float, sim_img_response[0].width, sim_img_response[0].height)
#PHI = lambda pixel: min(255.0, max(0, 75.0*math.log(pixel-0.5, math.e)))
PHI = lambda pixel: max(0, 65 * math.log(abs(pixel-0.7), math.e))

# apply PHI to each pixel
for row_idx in range(0, img.shape[0]):
  for col_idx in range(0, img.shape[1]):
    img[row_idx][col_idx] = PHI(img[row_idx][col_idx])
        
#cv2.imwrite('depthPlannerMin255orMax0or80timeslnx-pt5.jpg', img)
cv2.imwrite('depthPlannerMin0orMax0or70timeslnabsxminuspt7.jpg', img)


sim_img_responses = client.simGetImages([airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Scene,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.DepthPlanner,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.DepthPerspective,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.DepthVis,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.DisparityNormalized,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Segmentation,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.SurfaceNormals,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Infrared,
                                                                                      pixels_as_float=False,
                                                                                      compress=False),])

# raw imgs
img_names = ['Scene', 'DepthPlanner', 'DepthPerspective', 'DepthVis', 'DisparityNormalized', 'Segmentation', 'SurfaceNormals', 'Infrared']
for img_response_obj, img_name in zip(sim_img_responses, img_names):
  height = img_response_obj.height
  width = img_response_obj.width
  img = np.fromstring(img_response_obj.image_data_uint8, dtype=np.uint8).reshape(height, width, 4)
  cv2.imwrite('{}.jpg'.format(img_name), img)
  cv2.imwrite('grayscale_{}.jpg'.format(img_name), img)


# any experiments you'd want to do          
img_names = ['Scene', 'DepthPlanner', 'DepthPerspective', 'DepthVis', 'DisparityNormalized']

alter_pixel = lambda pixel_value: int( 255.0 * (1.0 / (1.0 + math.exp(-1.0 * (math.sqrt(2.71 * float(pixel_value)) - 4.5) ))) )
for img_response_obj, img_name in zip(sim_img_responses, img_names):
  height = img_response_obj.height
  width = img_response_obj.width
                   
  np_arr_of_img_2d = cv2.cvtColor(np.fromstring(img_response_obj.image_data_uint8, dtype=np.uint8).reshape(height, width, 4),
                                                                                          cv2.COLOR_BGR2GRAY)

  for row_idx in range(0, np_arr_of_img_2d.shape[0]):
    for col_idx in  range(0, np_arr_of_img_2d.shape[1]):
      np_arr_of_img_2d[row_idx][col_idx] = alter_pixel(np_arr_of_img_2d[row_idx][col_idx])
  
  cv2.imwrite('experiment_{}.jpg'.format(img_name), np_arr_of_img_2d)



