
import numpy as np
import time
from PIL import Image
import imutils
from imutils.video import VideoStream
import cv2
from keras.models import load_model

import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing import image

import warnings
warnings.filterwarnings('ignore')

from keras.preprocessing import image

from inception_resnet_v1 import *
import keras, keras_preprocessing

def preprocess_image(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

model = load_model('facenet_keras.h5')

def l2_normalize(x):
	return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
	euclidean_distance = source_representation - test_representation
	euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
	euclidean_distance = np.sqrt(euclidean_distance)
	return euclidean_distance




def face_check(model, face_pred, img2):
	img1_representation = face_pred
	img2_representation = l2_normalize(model.predict(preprocess_image(img2))[0,:])
 	
	euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
	print(euclidean_distance)

	threshold = 0.4
	if euclidean_distance < threshold:
		return True
	else:
		return False

#face_check(model, 'img1.jpg', 'img2.jpg')


mon = {'top': 160, 'left': 160, 'width': 160, 'height': 160}

vs = VideoStream(src=0)
time.sleep(2.0)

face =load_img('face1.jpg', target_size=(160, 160))
face = img_to_array(face)
face_pred = l2_normalize(model.predict(preprocess_image(face
))[0,:])

while True:
	print('Please enter a key to start check: ')
	input()
	vs.start()
	frame = vs.read()
	vs.stop()
	frame = cv2.resize(frame, (160,160))
	cv2.imshow('face', frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if face_check(model, face_pred, frame):
		print('Unlocked')
cv2.destroyAllWindows()
vs.stop()


