import numpy as np
import cv2, PIL
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
#import pandas as pd

frame = cv2.imread("examples/scene4.jpg")
#plt.figure()
#plt.imshow(frame)
#plt.show()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)


print(f"Index {0}, Arco:{ids[0]}")
print(f"Index {1}, Arco:{ids[1]}")
print(f"Index {2}, Arco:{ids[2]}")
print(f"Index {3}, Arco:{ids[3]}")
cv2.circle(frame_markers, (corners[0][0][2][0], corners[0][0][2][1]), 15, (0,0,255), -1)
cv2.circle(frame_markers, (corners[1][0][3][0], corners[1][0][3][1]), 15, (0,255,0), -1)
cv2.circle(frame_markers, (corners[2][0][1][0], corners[2][0][1][1]), 15, (255,0,0), -1)
cv2.circle(frame_markers, (corners[3][0][0][0], corners[3][0][0][1]), 15, (0,0,0), -1)
JJ = cv2.resize(frame_markers,(700,700))
cv2.imshow("1",JJ)

# plt.figure()
# plt.imshow(frame_markers)
# #for i in range(len(ids)):
#     #c = corners[i][0]
#     #plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
# plt.legend()
# plt.show()

cv2.waitKey()
cv2.destroyAllWindows()