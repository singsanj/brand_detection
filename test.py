from __future__ import division, print_function, absolute_import
import os,sys
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from constant import label
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse


img_prep = ImagePreprocessing()            
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()

network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.01)
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='model/logo-classifier.tfl.ckpt-5470')
model.load("model/logo-classifier.tfl.ckpt-5470")

def main():
	if len(sys.argv) > 1:
		test_image_fn = sys.argv[1]
		if not os.path.exists(test_image_fn):
			print("Not found:", test_image_fn)
			sys.exit(-1)
	print("Test image:", test_image_fn)
	img = scipy.ndimage.imread(test_image_fn, mode="RGB")
	img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
	prediction = model.predict([img])
	prediction_val = np.argmax(prediction[0])
	print('predicted logo" {}'.format(label(prediction_val)))

if __name__ == '__main__':
	main()

