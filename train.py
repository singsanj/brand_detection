from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import pickle
import numpy as np
from six.moves import range

def read_data():									
	with open("logo_dataset.pickle", 'rb') as f:
	    save = pickle.load(f)
	    X = save['train_dataset']					
	    Y = save['train_labels']					
	    X_test = save['test_dataset']				 
	    Y_test = save['test_labels']				
	    del save
   
	return [X, X_test], [Y, Y_test]

def reformat(dataset, labels):   
    dataset = dataset.reshape((-1, 32, 32,3)).astype(np.float32) 	
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)	 
    return dataset, labels

dataset, labels = read_data()
X,Y = reformat(dataset[0], labels[0])
X_test, Y_test = reformat(dataset[1], labels[1])
print('Training set', X.shape, Y.shape)
print('Test set', X_test.shape, Y_test.shape)            

X, Y = shuffle(X, Y)					

img_prep = ImagePreprocessing()    		
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512 , activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="model/logo-classifier.tfl.ckpt")
model.fit(X,Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=128, snapshot_epoch=True,
          run_id='logo-classifier')
model.save("logo-classifier.tfl")
print("Network trained and saved as logo-classifier.tfl!")
