# Persian numeral and alphabet recognition
This python project trains and employs two convolutional neural networks for recognizing Persian alphabet or numeral characters

Creating and training of the convolutional neural network is done using the Keras framework.


## Content  
1. [Phase one](#p1)
    1. [Form extraction](#p1.1)
2. [Phase two](#p2)
    1. [Dataset](#p2.1)
    2. [Training](#p2.2)
    3. [Detection](#p2.3)


## 1.Phase One <a name="p1"></a>
### 1.1 Form extraction <a name="p1.1"></a>
[Phase one](phaseOne.py) aims to extract predetermined boxes from the given [form](Form_A5.pdf).  
This process entails detecting and flattening the form from the input image. 
This is done using 4 Arucos and a perspective transform. Both using OpenCV.

## 2.Phase Two <a name="p2"></a>
### 2.1 Dataset preparation <a name="p2.1"></a>

In this phase, all students submitted two alphabet and digit samples in predefined forms.  
This [section](Dataset.ipynb) included flattening the photos and extracting and tagging alphabets and digits into two test and train datasets.  
The datasets are then used in the next step for training.

### 2.2 Training <a name="p2.2"></a>
Using the prepared images, two separate neural networks are [trained](Trainer.ipynb) for recognizing alphabets and digits.  
After training, a simple test is done using the test dataset for examining the network's accuracy.

### 2.3 Detection <a name="p2.3"></a>
[Phase two](phaseTwo.py) employs [Phase one](phaseOne.py) methods 
to extract images from the document, and using the saved neural network, recognizes the alphabet and digits separately, printing the end result.
