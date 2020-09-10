import numpy as np
import cv2
import Util_ARUCO as ut


image_path = "examples/scene1.jpg"
warped = ut.flatten_form(image_path, return_gray_scale=True)
student_no = ut.extract_box(warped, ut.dict_form_box["student_no"][0], ut.dict_form_box["student_no"][1], 8)
first_name = ut.extract_box(warped, ut.dict_form_box["first_name"][0], ut.dict_form_box["first_name"][1], 8)
last_name = ut.extract_box(warped, ut.dict_form_box["last_name"][0], ut.dict_form_box["last_name"][1], 8)
bachelor = ut.extract_box(warped, ut.dict_form_box["bch"][0], ut.dict_form_box["bch"][1], 1)[0]
master = ut.extract_box(warped, ut.dict_form_box["mst"][0], ut.dict_form_box["mst"][1], 1)[0]
phd = ut.extract_box(warped, ut.dict_form_box["phd"][0], ut.dict_form_box["phd"][1], 1)[0]

cv2.imwrite("output/output.jpg", warped)

for i in range(8):
    # student_no[i] = ut.normalize(student_no[i])
    # first_name[i] = ut.normalize(first_name[i])
    # last_name[i] = ut.normalize(last_name[i])
    cv2.imwrite("output/ID" + str(i) + ".jpg", student_no[i])
    cv2.imwrite("output/FN" + str(i) + ".jpg", first_name[i])
    cv2.imwrite("output/LN" + str(i) + ".jpg", last_name[i])

cv2.imwrite("output/PHD.jpg", phd)
cv2.imwrite("output/MS.jpg", master)
cv2.imwrite("output/BS.jpg", bachelor)
