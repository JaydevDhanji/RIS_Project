#!/usr/bin/env python3
import gym
import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv

class DuckDetector:
    def __init__(self, env_name="Duckietown-loop_obstacles-v0", max_steps=200):
        self.env = gym.make(env_name)
        self.max_steps = max_steps
        self.step_count = 0

        # Load duck template (grayscale)
        self.duck_template = cv2.imread("duck_template.png", 0)
        if self.duck_template is None:
            raise FileNotFoundError("⚠️ Missing 'duck_template.png'. Place a cropped duck image in this folder.")

        self.w, self.h = self.duck_template.shape[::-1]

    def detect_duck(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.duck_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.55   # adjust for sensitivity
        loc = np.where(res >= threshold)

        detected = False
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + self.w, pt[1] + self.h), (0, 0, 255), 2)
            detected = True

        return frame, detected

    def run(self):
        obs = self.env.reset()
        done = False

        while not done and self.step_count < self.max_steps:
            self.step_count += 1

            # Dummy action (straight drive)
            action = np.array([0.0, 0.3])
            obs, reward, done, info = self.env.step(action)

            # Detect ducks
            frame, detected = self.detect_duck(obs)

            if detected:
                print(f"Step {self.step_count}: 🦆 Duck detected!")

            # Show window (requires X11 forwarding or xvfb-run)
            cv2.imshow("Duck Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DuckDetector()
    detector.run()

