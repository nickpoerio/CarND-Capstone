# CarND Capstone

## Team
The 'KnightRider' team was created, but unfortunately no one joined, so I had to do it myself: Nicola Poerio (nickpoerio@hotmail.com)
The project works well on the simulator and should work on Carla.

## Implementation
Here's the description of the implementation, node by node.

### Waypoint Updater
This node subscribes to '/current\_pose' and '/base\_waypoints' topics and publishes 50 waypoints ahead of the vehicle to the 'final\_waypoints' topic at 20 Hz. A longer range would not only be unuseful, but would cause instability to the controller. Moreover, more points require additional computation. In order to work with the camera active I've set a 10 Hz rate to the node (faster rate is not necessary).

### DBW
This node subscribes to '/vehicle/dbw\_enabled', '/twist\_cmd' and 'current\_velocity' topics and publishes to '/vehicle/steering\_cmd', '/vehicle/throttle\_cmd' and '/vehicle/brake\_cmd' the inputs of steering, throttle and brake at a 50 Hz rate.

#### Twist Controller
This class invokes the yaw rate controller for the lateral direction and the pid controller for the longitudinal.
It resets the pid gains when the dbw is not active. I have excluded the use of low pass filter, as it delays the velocity signal, which is actually quite clean.

#### Yaw Controller
This class takes the desired velocity and angular velocity and compare them to the current one. I added a feedback term to the angular velocity error in order to improve the system stability. The base algorithm was in fact a simple inverse model based on vehicle kinematics.

### Waypoint Follower
I modified the Autoware C++ code in order to keep the lateral controller always active and provide stability to the system. Moreover I've set it to work at 50 Hz.

### Traffic Light Detector
This node subscribes to '/current\_pose', '/base\_waypoints', '/vehicle/traffic\_lights' and '/image\_color' topics. So it detects the closest upoming traffic light based on topics information and invokes the traffic light classifier for color understanding. I've set the rate to 10 Hz, both because camera information is published at this rate and because faster update cause excess computation.

### Traffic Light Classifier
This is the most original part of my code. Being the detection coming from other channels, and being myself interested much in end-to-end scene understanding and much less in image labeling ;) I decided to apply a CNN to classify the images: for this I decided to use the NVIDIA model used for the Behavioral Cloning project. The reason for this is that I wanted to understand the generalization capability of this architecture, which by the way is not that heavy as most popular end-to-end graphs presents in literature.
I obtained respectively 99.3% and 98.5% test accuracy on simulator and site images (about 4000), after just few epochs, which is quite remarkable. The code of my training is present in the light\_classifier folder. In order to properly work on the vehicle, I've set a probability threshold to 0.2 instead of 0.5, in order to solve classifier recall issues (precision is almost perfect).