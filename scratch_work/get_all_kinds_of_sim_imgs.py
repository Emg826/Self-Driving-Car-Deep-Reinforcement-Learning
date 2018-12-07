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

sim_img_responses = client.simGetImages([airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Scene,
                                                                                      pixels_as_float=True,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.DepthPlanner,
                                                                                      pixels_as_float=True,
                                                                                      compress=False),
                                                          airsim.ImageRequest(camera_name='0',
                                                                                      image_type=airsim.ImageType.Segmentation,
                                                                                      pixels_as_float=False,
                                                                                      compress=False)])

# raw imgs
alter_pixel = lambda pixel:  min(4096.0 / (pixel+11.0), 255.0)

img_names = ['Scene', 'DepthPlanner', 'Segmentation']
for img_response_obj, img_name in zip(sim_img_responses, img_names):
  height = img_response_obj.height
  width = img_response_obj.width

  img = None
  if img_name == 'DepthPlanner' or img_name == 'Scene':
    img = airsim.list_to_2d_float_array(img_response_obj.image_data_float, width, height)
    if img_name == 'Scene':
      print(img.shape)
      img = img * 255.0   # b4 this mult, all val in [0, 1]
      

    else:
      # apply pxiel wise filter
      for row_idx in range(0, img.shape[0]):
        for col_idx in  range(0, img.shape[1]):
          img[row_idx][col_idx] = alter_pixel(img[row_idx][col_idx])


  if img_name == 'Segmentation':
    #img = img * 255.0   # all vals 0 < .. < 1.0 otherwise
    img = np.fromstring(img_response_obj.image_data_uint8, dtype=np.uint8).reshape(height, width, 4)
    
  cv2.imwrite('{}.jpg'.format(img_name), img)

