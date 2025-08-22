#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/duck_detection_results/step_0050_detections.jpg')
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.title('Duck Detection Results')
plt.axis('off')
plt.show()
