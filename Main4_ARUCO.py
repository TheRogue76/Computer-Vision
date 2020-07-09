import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

dict_arco = {
    34: 0,
    35: 1,
    36: 2,
    33: 3,
}

I_orginal = cv2.imread("examples/scene1.jpg")

##ARUCO finding process
gray = cv2.cvtColor(I_orginal, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
# shows aurco ids
I_debug = aruco.drawDetectedMarkers(I_orginal.copy(), corners, ids)

source_points = np.zeros((4, 2), dtype=np.dtype('int32'))

####AURCO matching process
## source_points[0] = id: 34, croner: top left
## source_points[1] = id: 35, croner: top right
## source_points[2] = id: 36, croner: bot right
## source_points[3] = id: 33, croner: bot left
I_copy = I_orginal.copy()
for i in range(4):
    id = ids[i][0]
    corn = dict_arco[id]
    x = corners[i][0][corn][0]
    y = corners[i][0][corn][1]
    point = (x, y)
    source_points[corn][0] = x
    source_points[corn][1] = y
    # cv2.circle(I_copy, point, 10, (0, 0, 0), -1)

####destination image
## dest_points[0] = top left
## dest_points[1] = top right
## dest_points[2] = bot right
## dest_points[3] = bot left
dest_points = np.array([
    (0, 0),
    (500, 0),
    (500, 658),
    (0, 658),
])

H, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC, 4.0)
print(f"H:: {H}")
J_warped = cv2.warpPerspective(I_copy, H, (500, 658))
cv2.imshow("1", J_warped)
cv2.imwrite("output.jpg", J_warped)

studentNo = J_warped[212:254, 24:362]
studentNo = np.array_split(studentNo, 8, axis=1)
for i in range(8):
    cv2.imshow("student " + str(i), studentNo[i])

name = J_warped[269:311, 24:362]
name = np.array_split(name, 8, axis=1)
for i in range(8):
    cv2.imshow("name " + str(i), name[i])

familyName = J_warped[325:367, 24:362]
familyName = np.array_split(familyName, 8, axis=1)
for i in range(8):
    cv2.imshow("family " + str(i), familyName[i])

phd = J_warped[394:411, 45:62]
cv2.imshow("phd", phd)

masters = J_warped[394:411, 139:156]
cv2.imshow("masters", masters)

bachelor = J_warped[394:411, 280:296]
cv2.imshow("bachelor", bachelor)

cv2.waitKey()
cv2.destroyAllWindows()
