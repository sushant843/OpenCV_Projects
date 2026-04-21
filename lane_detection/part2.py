import json
import cv2
import numpy as np

with open("lane_detection\\test_label_new.json") as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    
    img = cv2.imread(data["raw_file"])
    cv2.imshow("Road Image", img)
    mask = np.zeros((720, 1280), dtype=np.uint8)

    for lane in data["lanes"]:
        points = []
        for x, y in zip(lane, data["h_samples"]):
            if x != -2:
                points.append((x, y))
        
        for i in range(len(points)-1):
            cv2.line(mask, points[i], points[i+1], 255, 5)

    cv2.imshow("mask", mask)
    cv2.waitKey(0)