import numpy as np
import cv2, PIL
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl

dict_arco = {
    36 : 2,
    33 : 3,
    35 : 1,
    34 : 0
}

frame = cv2.imread("examples/scene4.jpg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

points = np.zeros((4,2),dtype=np.dtype('int32'))

for i in range(4):
    id = ids[i][0]
    corn = dict_arco[id]
    x = corners[i][0][corn][0]
    y = corners[i][0][corn][1]
    point = (x,y)
    points[corn][0] = x
    points[corn][1] = y
    cv2.circle(frame_markers, point, 10, (0,0,0), -1)

dest_points = np.array([
    (0,0),
    (500,0),
    (500,500),
    (0,500),
])

H, mask = cv2.findHomography(points, dest_points, cv2.RANSAC,4.0)
print(f"H:: {H}")
J = cv2.warpPerspective(frame_markers,H, (500,500))
cv2.imshow("1",J)

cv2.waitKey()
#cv2.destroyAllWindows()