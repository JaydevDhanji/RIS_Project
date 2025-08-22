#!/usr/bin/env python3
import gym
import cv2
import numpy as np
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv

# Create the simulation environment
env = DuckietownEnv(
    map_name="loop_obstacles",
    domain_rand=False,
    draw_bbox=False
)
obs = env.reset()
#env.render()

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def detect_ducks(frame):
    """Detect yellow ducks in the image and draw bounding boxes"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2

    # Find contours of detected yellow areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    duck_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore small objects 
            continue

	x, y, w, h = cv2.boundingRect(cnt)
	aspect_ratio = w / float(h)

            # Filter out lane lines (very wide or very tall yellow shapes)
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:
                continue  # Likely a lane marking, skip

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "DUCK", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            duck_detected = True

    return frame, duck_detected

print("Press ESC to quit")

while True:
   action = np.array([0.0, 0.0])
    if key_handler[key.UP]:
        action[0] += 0.5
    if key_handler[key.DOWN]:
        action[0] -= 0.5
    if key_handler[key.LEFT]:
        action[1] += 1.0
    if key_handler[key.RIGHT]:
        action[1] -= 1.0
 
    obs, reward, done, info = env.step(action)  # Slow forward motion

    frame, duck_found = detect_ducks(obs)
    cv2.imshow("Duck Detection", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if duck_found:
        print("Duck detected ahead!")

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if done:
        env.reset()

env.close()
cv2.destroyAllWindows()

