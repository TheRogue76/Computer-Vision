import numpy as np
import cv2
import glob
from cv2 import aruco
import matplotlib.pyplot as plt


def flatten_photo(dict_arco, width, height, image_path):
    """flatten phto using aurco dictinary to given size
    dict_arco must follow as 
    {
        topleft  : 0
        topright : 1
        botright : 2
        botleft  : 3
    }

    Args:
        dict_arco (dict): arco dictonary
        width (int): width of warped picture
        height (int): height of warped picture
        image_path (string): image to read

    Returns:
        cv.image: warped image
    """
    I_orginal = cv2.imread(image_path)
    # ARUCO finding process
    gray = cv2.cvtColor(I_orginal, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    source_points = np.zeros((4, 2), dtype=np.dtype('int32'))
    if len(ids) != 4:
        return None
    # AURCO matching process
    # source_points[0] = corner: top left
    # source_points[1] = corner: top right
    # source_points[2] = corner: bot right
    # source_points[3] = corner: bot left
    for i in range(4):
        id = ids[i][0]
        corn = dict_arco[id]
        x = corners[i][0][corn][0]
        y = corners[i][0][corn][1]
        point = (x, y)
        source_points[corn][0] = x
        source_points[corn][1] = y

    # destination image
    # dest_points[0] = top left
    # dest_points[1] = top right
    # dest_points[2] = bot right
    # dest_points[3] = bot left
    dest_points = np.array([
        (0, 0),
        (width, 0),
        (width, height),
        (0, height), ])

    H, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC, 4.0)
    J_warped = cv2.warpPerspective(I_orginal, H, (width, height))
    return J_warped


dict_arco = {
    30: 0,
    31: 1,
    33: 2,
    32: 3,
}

size = 32  # size of each cell (pixel)
with_no = 14  # number of cells in row/width
height_no = 21  # number of cells in collumn/height

list_items_1 = [
    ["no_0", 10, "۰"],
    ["no_1", 10, "۱"],
    ["al_al", 14, "ا"],
    ["al_be", 14, "ب"],
    ["al_pe", 14, "پ"],
    ["al_te", 14, "ت"],
    ["al_sn", 14, "ث"],
    ["al_jm", 14, "ج"],
    ["al_ch", 14, "چ"],
    ["al_hh", 14, "ح"],
    ["al_kh", 14, "خ"],
    ["al_dl", 14, "د"],
    ["al_zl", 14, "ذ"],
    ["al_rr", 14, "ر"],
    ["al_zz", 14, "ز"],
    ["al_jz", 14, "ژ"],
    ["al_sn", 14, "س"],
    ["al_shn", 14, "ش"],
    ["al_sd", 14, "ص"],
    ["no_2", 10, "۲"],
    ["no_3", 10, "۳"]]

list_items_2 = [
    ["no_4", 10, "۴"],
    ["no_5", 10, "۵"],
    ["al_zd", 14, "ض"],
    ["al_ta", 14, "ط"],
    ["al_za", 14, "ظ"],
    ["al_ay", 14, "ع"],
    ["al_ghy", 14, "غ"],
    ["al_fe", 14, "ف"],
    ["al_gh", 14, "ق"],
    ["al_kf", 14, "ک"],
    ["al_gf", 14, "گ"],
    ["al_lm", 14, "ل"],
    ["al_mm", 14, "م"],
    ["al_nn", 14, "ن"],
    ["al_vv", 14, "و"],
    ["al_he", 14, "ه"],
    ["al_ye", 14, "ی"],
    ["no_6", 14, "۶"],
    ["no_7", 14, "۷"],
    ["no_8", 10, "۸"],
    ["no_9", 10, "۹"]]
index = -1
fnames_1 = glob.glob("dataset/raw/*_1*.jpg")
fnames_2 = glob.glob("dataset/raw/*_2*.jpg")
fs = [fnames_1, fnames_2]
ls = [list_items_1, list_items_2]
for fnames, list_items in zip(fs, ls):
    for image_path in fnames:
        index = index + 1
        J_warped = flatten_photo(dict_arco, size*with_no, size*height_no, image_path)
        if(J_warped == None):
            continue
        # cv2.imshow("1", J_warped)
        # cv2.imwrite(f"output/output{index}.jpg", J_warped)
        # cv2.waitKey(10)
        for row in range(len(list_items)):
            y0 = int(row * size)
            y1 = int(y0 + size)
            tmp = list_items[row][1]
            x0 = int((-tmp / 2 - 7) * size)
            x1 = int((tmp / 2 + 7) * size)
            tmp_arr = J_warped[y0:y1, x0:x1]
            tmp_arr = np.array_split(tmp_arr, tmp, axis=1)
            for i in range(tmp):
                cv2.imwrite(
                    f"dataset/processed/{list_items[row][0]}_{index}_{i}.jpg", tmp_arr[i])

cv2.destroyAllWindows()
