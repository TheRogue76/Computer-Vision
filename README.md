# Persian numeral and alphabet recognition
This python project traines and employs two convolutional neural network for recognizing persian alphabet or numeral characters

creating and training of the convolutional neural network is done using Keras framework


## Content  
1. [Phase one](#p1)
    1. [Form extraction](#p1.1)
2. [Phase two](#p2)
    1. [Dataset](#p2.1)
    2. [Training](#p2.2)
    3. [Detection](#p2.3)


## 1.Phase One <a name="p1"></a>
### 1.1 Form extraction <a name="p1.1"></a>
[Phase one](phaseOne.py) aims to extract predetermined boxes from the given [form](Form_a5.pdf).  
This process entails detecting and flattening the form from the input image. 
this is done using 4 Arucos and a perspective transform. both using opencv.

## 2.Phase Two <a name="p2"></a>
### 2.1 Dataset preperation <a name="p2.1"></a>

//TODO

### 2.2 Training <a name="p2.2"></a>
Using the prepared images, two separate neural networks are [trained](Trainer.ipynb) 
and their configurtions are strored.  
After traning, a simple testis done using the test dataset for examining the networks accuracy.

### 2.3 Detection <a name="p2.3"></a>
[Phase two](phaseTwo.py) employs [Phase one](phaseOne.py) methods 
to extract images from the document and using the saved neural network recognizes the alphabet and digits seperatly, printing the end result.
