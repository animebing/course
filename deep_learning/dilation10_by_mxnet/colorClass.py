from __future__ import print_function, division
import numpy as np
import cv2

classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic      sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",        "motorcycle", "bicycle", "nothing"]
color = [[128,64,128], [244,35,232], [70,70,70], [102,102,156], [190,153,153], [153,153,153], [250,170,30], [220,220,0], [107,142,35], [152,251,152], [70,130,180], [220,20,60], [255,0,0], [0,0,142], [0,0,70], [0,60,100], [0,80,100], [0,0,230], [119,11,32], [0, 0, 0]]
color = np.array(color, dtype=np.uint8)

colorMap = np.zeros((2000, 2500), dtype=np.uint8)

for i in xrange(4):
    for j in xrange(5):
        colorMap[i*500:(i+1)*500, j*500:(j+1)*500] = i*5+j

colorMap = color[colorMap.ravel()].reshape(2000, 2500, 3)
colorMap = cv2.cvtColor(colorMap, cv2.COLOR_RGB2BGR)
for i in xrange(4):
    for j in xrange(5):
        cv2.putText(colorMap, classes[i*5+j], (60+j*500, 60+i*500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (225, 225, 225), 3)
cv2.imwrite("colorMap.png", colorMap)
