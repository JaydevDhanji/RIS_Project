#!/usr/bin/env python
import gym
import gym_duckietown
import numpy as np
import cv2
import time
from gym_duckietown.envs import DuckietownEnv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
#parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
#parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()


if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        #camera_rand=args.camera_rand,
        #dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(arg.env_name)
# --- PID Controller ---
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# --- Duckietown environment ---
#env = gym.make(args.env_name)
obs = env.reset()
dt = 0.1  # control step
env.render()

# PID parameters
pid = PID(Kp=0.8, Ki=0.0, Kd=0.2)
base_speed = 0.3

def process_image(image):
    """Return the horizontal offset of the line center from image center"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Look only at yellow mask
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    h,w = mask.shape
    crop = mask [int(h*0.6):, :]
    moments = cv2.moments(crop)
    if moments["m00"] == 0:
        cx = w // 2  # fallback to center
    else:
        cx = int(moments["m10"] / moments["m00"])
    
    error = (w // 2 - cx) / (w // 2)  # normalized [-1, 1]
    
    # For visualization
    line_img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cv2.circle(line_img, (cx, crop.shape[0]//2), 5, (0,0,255), -1)
    cv2.imshow("Line", line_img)
    cv2.waitKey(1)
    
    return error

# --- Main loop ---
try:
    while True:
        start_time = time.time()
        
        error = process_image(obs)
        omega = pid.control(error, dt)  # steering
        action = np.array([base_speed, omega])  # [linear_velocity, angular_velocity]
        
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            obs = env.reset()
        
        # maintain consistent dt
        time.sleep(max(0, dt - (time.time() - start_time)))
finally:
    env.close()
    cv2.destroyAllWindows()

