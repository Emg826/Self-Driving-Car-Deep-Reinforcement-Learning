
# Using Deep Reinforcement Learning to Drive a Simulated Car in a Simulated Environment


![Panoramia view the car sees](https://github.com/Emg826/Reinforcement-Learning-Project/blob/DRL_steering_control/imgs/sample_panorama.png)

This is the repository for my senior independent study at the College of Wooster. The topic for my senior I.S. is using a deep double Q network to develop an end-to-end autonomous vehicle system. This deep double Q network will do its learning in Microsoft's AirSim simulator. Aside from this simulator being open source and free, one of the main reasons it was chosen (over, say, the Udacity self-driving car simulator) was that this simulator offers diverse environments (urban and rural) with very dynamic elements (other cars, pedestrians, variable weather and lightling conditions, etc.). 

This simulator operates under the client-server model with the server being the simulation and the client being some program that determines how the car should be controlled. Fortunately, the simulator offers a Python client API, so Python can and will be used to implement the deep double Q network system. 

As of today, Wednesday, September 19, 2018, I have become decently familiar with the Python API. The two Python programs that I coded have gotten me to the point where I at least understand how to control the car from the Python client and how to request and use images from the simulation in the car controll process. The next step is to start developing a deep reinforcement car controller. The three main challenges that I expect to encounter in this next step are deteriming what signal should be back-propogated through the network, keeping the car in its lane, and writing the code for the deep reinforcement algorithm (in keras-rl).
