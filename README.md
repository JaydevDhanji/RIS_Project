**Gym Duckietown: Duck Detection & Lane Following**

This repository contains a Duckietown project using Docker containers and images to implement duck detection and lane following with PID control. It is designed for simulation in the Gym Duckietown environment.

**Features**

**-Duck Detection** – Detects duck objects in the Duckietown environment.
**-Lane Following** – Implements PID-based control to follow lanes accurately.
**-Dockerized Environment** – All dependencies are handled via Docker for easy setup.
**-Gym Duckietown Integration** – Seamless interaction with the Duckietown simulator.

**INSTALLATION:**
Clone the repository 
```bash
git clone https://github.com/JaydevDhanji/gym-duckietown.git
cd gym-duckietown

Build the Docker Image 
docker build -t duckietown-env .

Initiate xhost on your host laptop by running this line of code 
xhost +local:root

Run the Docker image 
docker run -it --rm duckietown-env

**USAGE:** 
For the Duck Detection with Keyboard Control, run this line of code
./trial1.py --env-name Duckietown-loop_obstacles-v0

For the Lane Following with PID Control, run this line of code 
./lane4.py 

When you run the lane following, you will get 2 options, the first one is to test without visualizing it and the second one is running it on the simulator. 


