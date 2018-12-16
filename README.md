
# Using Deep Reinforcement Learning to Drive a Simulated Car

![A roughly 135 degree view of what is front of the car. This is what the car sees but in grayscale.](https://github.com/Emg826/Reinforcement-Learning-Project/blob/master/imgs/scene_ifelse.jpg?raw=true)

The objective of this project is to use deep reinforcement learning to create a self-driving car system in Microsoft's self-driving car simulator, AirSim, that can simultaneously steer and manage speed. The specific algorithm in use here is double deep Q-learning which uses double deep Q-networks (DDQNs). As of today, December 16, 2018, I am working on the steering-end of the project, which, I am just about finished with. Once the Spring semester begins, the plan is to let the network train on a more powerful computer at the college that I attend, the College of Wooster. Given the recent literature, the general principle that deep learning requires a lot of data in order to do its learning, and the fact that reinforcement learning is more or less "learning by doing" or "learning by trial and error," training will likely take a substantial amount of time. 


## DDQN Inputs
The DDQN that I am currrently using has been configured to accept multiple inputs: images and numerical sensor data. There are two kinds of images that the neural network sees: regular images (like the one at the top of this document), and a depth planner image (like the one directly below). The depth planner image functions as a kind of depth sensor, e.g., a proximity sensor or ultrasonic radar, in this project. Pure white means that an object is within 5 feet of the vehicle, grays indicate that an object is within the depth sensor's range, and black represents emptiness or something that is outside of the depth sensor's range. Note that both kinds of images have been stitched together so as to provide a wider field of view, an approximately 135 degree field of view.

![Think of this as a sonar scan of the car's environment.](https://github.com/Emg826/Reinforcement-Learning-Project/blob/master/imgs/depth_planner_ifelse.jpg?raw=true)

The numerical sensor data that the network takes in are: 
1-2. GPS (x, y) coordinates of car
3. Manhattan distance from destination point (x, y)
4. Yaw/compass direction of car in radians
5. Bearing (relative to the destination)
6. Current steering angle
7-8. Angular velocity (x, y)
9-10. Linear acceleration (x, y)
11-12. Angular acceleration (x, y)
13. Speed
14-15. Absolute difference from current location to destination for x and y each
16-17. (x, y) coordinates of destination.
    
    
## Multi-input DDQN Architecture
Given all of this data, the DDQN is supposed to be able to learn how to steer itself so as to arrive at some destination while travelling at a low speed (<20mph). The architecture of this network can be seen below.



![The architecture of the current DDQN I am using; it takes 3 inputs: sensor data, camera images, and depth camera images.](https://github.com/Emg826/Reinforcement-Learning-Project/blob/master/imgs/multi-input-ddqn.png?raw=true)
