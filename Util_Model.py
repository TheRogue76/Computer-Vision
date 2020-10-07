import warnings

warnings.filterwarnings('ignore')
import numpy as np
import cv2
from cv2 import aruco
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
import Util_ARUCO as ut
import cv2
from cv2 import aruco

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


# model building processes
def build_model(inputs, num_classes):
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


def load_model(file_name, num_classes):
    """
    build and compiles model using num_classes
    and loads weights from specified file
    """
    input = Input((28, 28, 1))
    model = build_model(input, num_classes)
    opt = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
    # load model from file
    model.load_weights(file_name)
    return model


def run_predict(model, single_image):
    image = img_to_array(single_image) / 255.
    image.copy()
    image = np.expand_dims(image, 0)
    # run prediction
    predictions = model.predict(image)[0]
    label = np.argmax(predictions)
    proba = np.max(predictions)
    return (label, proba)
