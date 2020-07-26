import warnings

warnings.filterwarnings('ignore')
import numpy as np
import cv2
import Util_ARUCO as ua
import Util_Model as um
from operator import itemgetter


def isBoxEmpty(image, group):
    # ret, grey = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
    # histg = cv2.calcHist([grey], [0], None, [256], [0, 256])
    # if histg[255] > histg[0]:
    #     return True
    # else:
    #     return False
    return False


image_path = "examples/scene1.jpg"
warped = ua.flatten_form(image_path, return_gray_scale=True)
student_no = ua.extract_box(warped, ua.dict_form_box["student_no"][0], ua.dict_form_box["student_no"][1], 8)
first_name = ua.extract_box(warped, ua.dict_form_box["first_name"][0], ua.dict_form_box["first_name"][1], 8)
last_name = ua.extract_box(warped, ua.dict_form_box["last_name"][0], ua.dict_form_box["last_name"][1], 8)
bachelor = ua.extract_box(warped, ua.dict_form_box["bch"][0], ua.dict_form_box["bch"][1], 1)[0]
master = ua.extract_box(warped, ua.dict_form_box["mst"][0], ua.dict_form_box["mst"][1], 1)[0]
phd = ua.extract_box(warped, ua.dict_form_box["phd"][0], ua.dict_form_box["phd"][1], 1)[0]
# standardize the images
for i in range(8):
    student_no[i] = ua.standardize(student_no[i])
    first_name[i] = ua.standardize(first_name[i])
    last_name[i] = ua.standardize(last_name[i])

# compile and prepare model
model_num = um.load_model("model_no.h5", 10)
model_alp = um.load_model("model_alp.h5", 32)

# predict student number
predictedNumber = []
numberProbability = []
for i, test in enumerate(student_no):
    label, proba = um.run_predict(model_num, test)
    name_label = um.list_dataset_items[label][2]
    predictedNumber.append(name_label)
    numberProbability.append(proba)

# predict name
predictedName = []
nameProbability = []
for i, test in enumerate(first_name):
    if not isBoxEmpty(test, "name"):
        label, proba = um.run_predict(model_alp, test)
        label = label + 10
        name_label = um.list_dataset_items[label][2]
        predictedName.append(name_label)
        nameProbability.append(proba)
    else:
        continue

# predict family name
predictedFamilyName = []
familyNameProbability = []
for i, test in enumerate(last_name):
    if not isBoxEmpty(test, "familyName"):
        label, proba = um.run_predict(model_alp, test)
        label = label + 10
        name_label = um.list_dataset_items[label][2]
        predictedFamilyName.append(name_label)
        familyNameProbability.append(proba)
    else:
        continue


options = ["bachelor", "master", "phd"]
optionsAvg = np.array([np.average(bachelor), np.average(master), np.average(phd)])

print(predictedNumber)
print(numberProbability)
print(list(reversed(predictedName)))
print(list(reversed(nameProbability)))
print(list(reversed(predictedFamilyName)))
print(list(reversed(familyNameProbability)))
print(options[np.argmin(options)])
