import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

dict_aruco_form = {
    34: 0,
    35: 1,
    36: 2,
    33: 3,
}

dict_form_box = {
    "student_no": [(24, 212), (362, 254)],
    "first_name": [(24, 269), (362, 311)],
    "last_name": [(24, 325), (362, 367)],
}

studentNo = J_warped[212:254, 24:362]
name = J_warped[269:311, 24:362]
familyName = J_warped[325:367, 24:362]


def flatten_photo(image_path, dict_aruco_photo, width, height):
    """flatten photo using aruco dictionary to given size
    dict_aruco must follow as
    {
        topleft  : 0
        topright : 1
        botright : 2
        botleft  : 3
    }

    Args:
        dict_aruco_photo (dict): aruco dictionary
        width (int): width of warped picture
        height (int): height of warped picture
        image_path (string): image to read

    Returns:
        cv.image: warped image
    """
    try:
        img = cv2.imread(image_path)
        # ARUCO finding process
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        source_points = np.zeros((4, 2), dtype=np.dtype('int32'))
        if len(ids) != 4:
            print("Less than 4 aruco", image_path)
            return None
        # Fill source_points from corner of aruco
        for i in range(4):
            aruco_id = ids[i][0]
            corn = dict_aruco_photo[aruco_id]
            x = corners[i][0][corn][0]
            y = corners[i][0][corn][1]

            source_points[corn][0] = x
            source_points[corn][1] = y
            pass
        dest_points = np.array([
            (0, 0),  # Top left
            (width, 0),  # Top Right
            (width, height),  # Bot Right
            (0, height), ])  # Bot Left
        # calculate Homography and warp image
        H, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC, 4.0)
        warped = cv2.warpPerspective(img, H, (width, height))
    except:
        print(f"ERROR: path:{image_path}")
        return None
    return warped


def flatten_form(image_path, width=500, height=685):
    warped = flatten_photo(image_path, dict_aruco_form, width, height)
    return warped
    pass


def extract_box(image, point_top_left, point_bottom_right, pieces):
    """Extracts the specified box from image and cuts in
    equal pieces along x axis

    Args:
        image (dict): aruco dictionary
        point_top_left ((int,int)): tuple of x,y
        point_bottom_right ((int,int)): tuple of x,y
        pieces (int): number of pieces in x-axis

    Returns:
        Array of cv.image: cut out pieces
    """
    x_left = point_top_left[0]
    x_right = point_bottom_right[0]
    y_top = point_top_left[1]
    y_bottom = point_bottom_right[1]
    box = image[y_top:y_bottom, x_left:x_right]
    cut = np.array_split(box, 8, axis=1)
    pass


def normalize(image):
    """
    normalizes images histogram
    """
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255


studentNo = J_warped[212:254, 24:362]
studentNo = np.array_split(studentNo, 8, axis=1)
for i in range(8):
    cv2.imwrite("output/ID" + str(i) + ".jpg", studentNo[i])

name = J_warped[269:311, 24:362]
name = np.array_split(name, 8, axis=1)
for i in range(8):
    cv2.imwrite("output/FN" + str(i) + ".jpg", name[i])

familyName = J_warped[325:367, 24:362]
familyName = np.array_split(familyName, 8, axis=1)
for i in range(8):
    cv2.imwrite("output/LN" + str(i) + ".jpg", familyName[i])

phd = J_warped[394:411, 45:62]
cv2.imwrite("output/PHD.jpg", phd)

masters = J_warped[394:411, 139:156]
cv2.imwrite("output/MS.jpg", masters)

bachelor = J_warped[394:411, 280:296]
cv2.imwrite("output/BS.jpg", bachelor)

cv2.waitKey()
cv2.destroyAllWindows()
