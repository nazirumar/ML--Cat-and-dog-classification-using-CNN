from os import PathLike
import numpy
from numpy.core.records import array
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import  image
import numpy as np












































# Model Path
mymodel_Path =r'C:\Users\NAZBEEN_MULTIMEDIA\Documents\python_project\Computer_Vision\Cat_Dog\Cats-dogs.h5'

# Load Model Path
new_model = tf.keras.models.load_model(mymodel_Path)
Img_height = 200
Img_width = 200


# Load Testing Imge
img = r'C:\Users\NAZBEEN_MULTIMEDIA\Documents\python_project\Computer_Vision\Cat_Dog\test\dog\dog (36).jpg'
img =image.load_img(img, target_size=(Img_width, Img_height))
# Image To array
img = tf.keras.utils.img_to_array(img)
# np.true_divide(img, 255)
img = tf.expand_dims(img, axis=0)
prediction = new_model.predict(img)
predict = np.argmax(prediction, axis=-1)
if predict == 0:
    print("Cat")
elif predict == 1:
    print('Dog')
else:
    predict