import random
import cv2
import numpy as np
from track import Track

track = cv2.imread("IMG_20211026_200255.jpg")
track = cv2.copyMakeBorder(track, 0, 1000, 1500, 1000, cv2.BORDER_CONSTANT, None, value=0)


t = Track(track, np.array([(4650, 755), (7230, 3750), (88, 4220), (3270, 715)], np.int32), [9, 5])
for index in range(len(t.types)):
    t.set_type(random.randint(1, 10), index)

data = {}
for type in range(1, 11):
    data[type] = [100, random.randint(1, 100)]
    data[type].append(data[type][1]/data[type][0])

heat_map = t.generate_heat_map(data)

cv2.imshow("map", t.map)

cv2.imwrite("heatmap.jpg", heat_map)
cv2.imwrite("output.jpg", t.map)
cv2.imwrite("output1.jpg", track)
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.waitkey(1)
