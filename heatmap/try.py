import cv2
import numpy as np
from transform import four_point_transform

track = cv2.imread("IMG_20211026_200255.jpg")
track = cv2.copyMakeBorder(track, 0, 1000, 1500, 1000, cv2.BORDER_CONSTANT, None, value = 0)

#track = cv2.resize(track, (1280,720), interpolation = cv2.INTER_AREA)

transform = four_point_transform(track,np.array([(4650,755),(7230,3750),(88,4220),(3270,715)], np.int32),[5,9])
dims = transform.shape[:2]
y_increment = dims[0]//9
x_increment = dims[1]//5

y = 0
x = 0
while True:
    y += y_increment
    if y < dims[0]:
        cv2.line(transform, (0,y), (dims[1],y), (0,255,0), 2)
    else: break

while True:
    x += x_increment
    if x < dims[1]:
        cv2.line(transform, (x,0), (x,dims[0]), (0,255,0), 2)
    else: break

cv2.imshow("try",transform)
cv2.imwrite("output.jpg", transform)
cv2.imwrite("output1.jpg", track)
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.waitkey(1)