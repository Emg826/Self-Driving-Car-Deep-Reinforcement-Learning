import numpy as np
import time

class ProximitySensor():
  """
  Class to take a depth planner image and convert it into a proximity sensor
  output, which I define to be an array with num_sectors sectors.
  Fundamentally, it works by summing up the pixels in the depth image's columns.
  Can think of it like a histogram but for the pixels in the depth planner image.
  """
  def __init__(self, max_distance=12.0, kill_distance=3.0, num_proximity_sectors=32):
    """
    :param max_distance: beyond this distance (in sim. meters) the sensor will
    not pick up on things - proximity = 0 (min value)
    :param kill_distance: everything in the circle with this distance will be given the highest
    proximity value --> indicates collision imminent - proximity = 1 (max value)
    :param num_proximity_sectors: number of columns of width: img_width / num_proximity_sectors
    # ^ https://www.blocklayer.com/circle-dividereng.aspx  -- keep in mind the FOV of the image, though

    """
    self.max_distance = max_distance
    self.kill_distance = kill_distance
    self.num_proximity_sectors = num_proximity_sectors
    self.max_proximity_value = 1.0
    self.min_proximity_value = 0.0

    # will apply a piecewise function: y = { max_proximity_value . if x < kll_distance; 0 . if x > max_distance; mx+b if kill_distance <= x <= max_distance }
    # m = (y2-y1) / (x2-x1)  # http://www.webmath.com/_answer.php
    self.slope_of_line_outside_kill_distance = (self.min_proximity_value - self.max_proximity_value) / (self.max_distance - self.kill_distance)
    assert self.slope_of_line_outside_kill_distance <= 0
    
    # b = y - mx
    self.intercept_of_line_outside_kill_distance = 0.0 - self.slope_of_line_outside_kill_distance * self.max_distance

    self.outside_kill_inside_max_proximity = lambda depth_planner_pixel: self.slope_of_line_outside_kill_distance * depth_planner_pixel + self.intercept_of_line_outside_kill_distance
    
    # piecewise function - as a lambda function so as to vectorize (approx x4 faster)
    self.depth_planner_pixel_to_proximity_pixel = lambda depth_planner_pixel: self.max_proximity_value if depth_planner_pixel < self.kill_distance  \
                                                                                                                                        else (self.min_proximity_value if depth_planner_pixel > self.max_distance \
                                                                                                                                                                                else self.outside_kill_inside_max_proximity(depth_planner_pixel))
    # a function that'll loop over each pixel in image - self.depth_planner_image_to_proximity_image(2d_depth_planner_img_numpy_array)
    self.depth_planner_image_to_proximity_image = np.vectorize(self.depth_planner_pixel_to_proximity_pixel)

  def depth_planner_image_to_proximity_list(self, depth_planner_image):
    """
    Take in a depth planner image; output a list of length num_proximity_sectors
    :param depth_planner_image: 2D numpy array; airsim.ImageTypes.DepthPlanner
    """
    proximity_image = self.depth_planner_image_to_proximity_image(depth_planner_image)
    
    assert proximity_image.shape == depth_planner_image.shape

    # get indicies for each sector's beginning
    sector_start_stop_col_indices = np.linspace(0, depth_planner_image.shape[1], self.num_proximity_sectors+1, endpoint=True, dtype=np.int)

    proximities_by_sector = []
    for sector_count in range(0, self.num_proximity_sectors):
      sector_start_idx = sector_start_stop_col_indices[sector_count]
      sector_end_idx = sector_start_stop_col_indices[sector_count+1]

      proximities_by_sector.append(np.sum(proximity_image[:,sector_start_idx:sector_end_idx])) # all rows, cols in his sector

    assert len(proximities_by_sector) == self.num_proximity_sectors

    #print(proximities_by_sector)
    return proximities_by_sector
    
