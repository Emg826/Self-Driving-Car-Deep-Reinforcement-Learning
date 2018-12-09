import cv2
import numpy as np
from matplotlib import pyplot as plt

airsim_scene_img_fp = 'imgs/Scene.jpg'
airsim_depth_planner_img_fp = 'imgs/DepthPlanner.jpg'

def input_img(img_fp):
  img = cv2.imread(img_fp)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  return img

scene_img = input_img(airsim_scene_img_fp)
depth_planner_img = input_img(airsim_depth_planner_img_fp)

print(scene_img.shape)
print(depth_planner_img.shape)

"""
plt.subplot(121)
plt.imshow(scene_img)

# this is actually good at detecting lines on road and curb
plt.subplot(122)
edge_detected_scene_img = cv2.Canny(scene_img, 250, 180)  # gaussian blur h, gaussian blur w?

print(edge_detected_scene_img.shape)
overlay_edge_on_ground_truth_scene_img = cv2.addWeighted(edge_detected_scene_img, 0.55,
                                                   scene_img, 0.45, 1)
plt.imshow(overlay_edge_on_ground_truth_scene_img)
plt.show()



# see grid lines to get an idea of how large convolution filters actually are
plt.subplot(211)
plt.imshow(overlay_edge_on_ground_truth_scene_img)
"""



# print(overlay_edge_on_ground_truth_scene_img.shape)  # (124, 350)
"""
kernel = (12, 10)
kernel_h = kernel[0]
kernel_w = kernel[1]

plt.xticks(np.arange(0, overlay_edge_on_ground_truth_scene_img.shape[1], kernel_w))
plt.yticks(np.arange(0, overlay_edge_on_ground_truth_scene_img.shape[0], kernel_h))


plt.grid(b=True, which='major', axis='both', linewidth=1.0)
plt.show()


overlay_edge_on_ground_truth_scene_img = cv2.addWeighted(cv2.addWeighted(edge_detected_scene_img, 0.6,
                                                                                                             scene_img, 0.4, 1),
                                                                                    0.5,
                                                                                    depth_planner_img,
                                                                                    0.5, 1)
plt.imshow(overlay_edge_on_ground_truth_scene_img, cmap='gray')
plt.show()
"""


kernel = (25, 25)
kernel_h = kernel[0]
kernel_w = kernel[1]

plt.xticks(np.arange(0, scene_img.shape[1], kernel_w))
plt.yticks(np.arange(0, scene_img.shape[0], kernel_h))
plt.grid(b=True, which='major', axis='both', linewidth=1.0)


edge_detected_scene_img = cv2.Canny(scene_img, 250, 180)  # gaussian blur h, gaussian blur w?
overlay_edge_on_ground_truth_scene_img = cv2.addWeighted(edge_detected_scene_img, 0.6,
                                                                                     scene_img, 0.40, 1)
plt.subplot(221)
plt.xticks(np.arange(0, scene_img.shape[1], kernel_w))
plt.yticks(np.arange(0, scene_img.shape[0], kernel_h))
plt.grid(b=True, which='major', axis='both', linewidth=1.0, alpha=0.5)
plt.imshow(scene_img, cmap='gray')


plt.subplot(222)
plt.xticks(np.arange(0, scene_img.shape[1], kernel_w))
plt.yticks(np.arange(0, scene_img.shape[0], kernel_h))
plt.grid(b=True, which='major', axis='both', linewidth=1.0, alpha=0.5)
plt.imshow(edge_detected_scene_img, cmap='gray')


kernel = (18, 18)
kernel_h = kernel[0]
kernel_w = kernel[1]
plt.subplot(223)
plt.xticks(np.arange(0, scene_img.shape[1], kernel_w))
plt.yticks(np.arange(0, scene_img.shape[0], kernel_h))
plt.grid(b=True, which='major', axis='both', linewidth=1.0, alpha=0.5)
plt.imshow(depth_planner_img, cmap='gray')


plt.subplot(224)
plt.xticks(np.arange(0, scene_img.shape[1], kernel_w))
plt.yticks(np.arange(0, scene_img.shape[0], kernel_h))
plt.grid(b=True, which='major', axis='both', linewidth=1.0, alpha=0.5)
plt.imshow(overlay_edge_on_ground_truth_scene_img, cmap='gray')

plt.show()
