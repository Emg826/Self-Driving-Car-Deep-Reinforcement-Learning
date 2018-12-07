import cv2
import numpy as np
from matplotlib import pyplot as plt

airsim_img_fp = 'airsim.jpg'

img = cv2.imread(airsim_img_fp, 0)
img = img[int(img.shape[0] * 0.525):,:]

plt.subplot(121)
plt.imshow(img)

# this is actually good at detecting lines on road and curb
plt.subplot(122)
edge_detected_img = cv2.Canny(img, 250, 180)  # gaussian blur h, gaussian blur w?
overlay_edge_on_ground_truth_img = cv2.addWeighted(edge_detected_img, 0.65,
                                                   img, 0.35, 1)
plt.imshow(overlay_edge_on_ground_truth_img, cmap='gray')
plt.show()


# see grid lines to get an idea of how large convolution filters actually are
plt.subplot(211)
plt.imshow(overlay_edge_on_ground_truth_img)

# print(overlay_edge_on_ground_truth_img.shape)  # (124, 350)
kernel = (7, 9)
kernel_h = kernel[0]
kernel_w = kernel[1]

plt.xticks(np.arange(0, overlay_edge_on_ground_truth_img.shape[1], kernel_w))
plt.yticks(np.arange(0, overlay_edge_on_ground_truth_img.shape[0], kernel_h))


plt.grid(b=True, which='major', axis='both', linewidth=1.0)
plt.show()
