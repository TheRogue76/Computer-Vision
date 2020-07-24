import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import keras
import glob
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt

#model building processes
def build_model(inputs):
  x = inputs

  x = Conv2D(filters=20, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

  x = Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x) 

  x = Flatten()(x)
  x = Dense(500, activation="relu")(x)
  outputs = Dense(num_classes, activation="softmax")(x)

  model = Model(inputs, outputs, name="LeNet")
  model.summary()
  
  return model

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

studentNo = J_warped[212:254, 24:362]
studentNo = np.array_split(studentNo, 8, axis=1)
for i in range(8):
    cv2.imwrite("output/ID"+str(i)+".jpg", studentNo[i])

name = J_warped[269:311, 24:362]
name = np.array_split(name, 8, axis=1)
for i in range(8):
    cv2.imwrite("output/FN" + str(i) + ".jpg", name[i])

familyName = J_warped[325:367, 24:362]
familyName = np.array_split(familyName, 8, axis=1)
for i in range(8):
    cv2.imwrite("output/LN" + str(i) + ".jpg", familyName[i])
    gray = cv2.cvtColor(familyName[i], cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    histg = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if histg[255] > histg[0]:
        print("empty")
    else:
        print("not empty")

phd = J_warped[394:411, 45:62]
cv2.imwrite("output/PHD.jpg", phd)

masters = J_warped[394:411, 139:156]
cv2.imwrite("output/MS.jpg", masters)

bachelor = J_warped[394:411, 280:296]
cv2.imwrite("output/BS.jpg", bachelor)

list_dataset_items = [
	["no_0_", 0, "۰"],
    ["no_1_", 1, "۱"],
    ["no_2_", 2, "۲"],
    ["no_3_", 3, "۳"],
    ["no_4_", 4, "۴"],
    ["no_5_", 5, "۵"],
    ["no_6_", 6, "۶"],
    ["no_7_", 7, "۷"],
    ["no_8_", 8, "۸"],
    ["no_9_", 9, "۹"],
    ["al_al_", 10, "ا"],
    ["al_be_", 11, "ب"],
    ["al_pe_", 12, "پ"],
    ["al_te_", 13, "ت"],
    ["al_ss_", 14, "ث"],
    ["al_jm_", 15, "ج"],
    ["al_ch_", 16, "چ"],
    ["al_hh_", 17, "ح"],
    ["al_kh_", 18, "خ"],
    ["al_dl_", 19, "د"],
    ["al_zl_", 20, "ذ"],
    ["al_rr_", 21, "ر"],
    ["al_zz_", 22, "ز"],
    ["al_jz_", 23, "ژ"],
    ["al_sn_", 24, "س"],
    ["al_shn_", 25, "ش"],
    ["al_sd_", 26, "ص"],
    ["al_zd_", 27, "ض"],
    ["al_ta_", 28, "ط"],
    ["al_za_", 29, "ظ"],
    ["al_ay_", 30, "ع"],
    ["al_ghy_", 31, "غ"],
    ["al_fe_", 32, "ف"],
    ["al_gh_", 33, "ق"],
    ["al_kf_", 34, "ک"],
    ["al_gf_", 35, "گ"],
    ["al_lm_", 36, "ل"],
    ["al_mm_", 37, "م"],
    ["al_nn_", 38, "ن"],
    ["al_vv_", 39, "و"],
    ["al_he_", 40, "ه"],
    ["al_ye_", 41, "ی"]
]
list_dataset_items = np.array(list_dataset_items)
#compile and prepare model
input = Input((28, 28, 1))
model_alp = build_model(input)
model_num = build_model(input)
opt = Adam()
model_alp.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
model_num.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
#load model from file
model_alp.load_weights("model_alp.h5")
model_num.load_weights("model_no.h5")

#predict student number
for i, test in enumerate(studentNo):
    #prepare image for test (normalize to 0-1, and make batch)
    image = img_to_array(test) / 255.
    orig_img = image.copy()
    image = np.expand_dims(image, 0)
    #run prediction
    predictions = model_num.predict(image)[0]
    label = np.argmax(predictions)
    #get predicted label character
    name_label = list_dataset_items[label][2]
    proba = np.max(predictions)

#predict name
for i, test in enumerate(name):
    #prepare image for test (normalize to 0-1, and make batch)
    image = img_to_array(test) / 255.
    orig_img = image.copy()
    image = np.expand_dims(image, 0)
    #run prediction
    predictions = model_alp.predict(image)[0]
    label = np.argmax(predictions)
    #re-offset labels because working with alphabets
    label = label+10
    #get predicted label character
    name_label = list_dataset_items[label][2]
    proba = np.max(predictions)
    
#predict family name
for i, test in enumerate(familyName):
    #prepare image for test (normalize to 0-1, and make batch)
    image = img_to_array(test) / 255.
    orig_img = image.copy()
    image = np.expand_dims(image, 0)
    #run prediction
    predictions = model_alp.predict(image)[0]
    label = np.argmax(predictions)
    #re-offset labels because working with alphabets
    label = label+10
    #get predicted label character
    name_label = list_dataset_items[label][2]
    proba = np.max(predictions)