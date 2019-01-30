'''
Before running this file bounding boxes must be processed using
process_bounding_boxes_bdd.py

This file takes the processed bounding boxes and does the following:
    1. Computes the pedestrians velocity and prints the error under the assumption
    that the pedestrian will maintain their veloicty
    2. Computes E_x and E_y. This is the value to be predicted by the model using
    the constant veloicty correction term. C_x = 1 - E_x
    3. Splits the data training and validation
'''
import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
from scipy.misc import imresize
import scipy.ndimage
import sys
import processing_utils as utils

PATH = '../data/'
MIN_LENGTH_PAST = 10
MIN_LENGTH_FUTURE= 15
VELOCITY_FRAMES = 5


features = pd.read_pickle(PATH + 'bdd_10k_location_features.pkl')
features = features[features['Labeled']==1]

features['Final_x'] = features['Future_x'].apply(lambda x: x[-1])
features['Final_y'] = features['Future_y'].apply(lambda y: y[-1])

##################### Compute velocity and make predictions #####################
print('Computing velocity...')

features['Velocity_x'] = features['Past_x'].apply(lambda x: utils.mean_velocity(x,VELOCITY_FRAMES))
features['Velocity_y'] = features['Past_y'].apply(lambda y: utils.mean_velocity(y,VELOCITY_FRAMES))

features['Predicted_x'] = features['Mid_x'] + (MIN_LENGTH_FUTURE * features['Velocity_x'])
features['Predicted_y'] = features['Mid_y'] + (MIN_LENGTH_FUTURE * features['Velocity_y'])

print('Getting predictions...')

features['Predicted_x_seq'] = features.apply(lambda x: utils.get_seq_preds(x['Mid_x'],x['Velocity_x'],MIN_LENGTH_FUTURE),axis=1 )
features['Predicted_y_seq'] = features.apply(lambda x: utils.get_seq_preds(x['Mid_y'],x['Velocity_y'],MIN_LENGTH_FUTURE),axis=1 )

features['E_x'] = features.apply(lambda x: (x['Future_x']-x['Predicted_x_seq']),axis=1 )
features['E_y'] = features.apply(lambda y: (y['Future_y']-y['Predicted_y_seq']),axis=1 )

##################### Get errors #####################

print('Computing errors...')
features['MSE_15'] = features.apply(lambda x: utils.calc_mse(x['Predicted_x_seq'],x['Future_x'],x['Predicted_y_seq'],x['Future_y'],15),axis=1 )
features['MSE_10'] = features.apply(lambda x: utils.calc_mse(x['Predicted_x_seq'],x['Future_x'],x['Predicted_y_seq'],x['Future_y'],10),axis=1 )
features['MSE_5'] = features.apply(lambda x: utils.calc_mse(x['Predicted_x_seq'],x['Future_x'],x['Predicted_y_seq'],x['Future_y'],5),axis=1 )

features['EPE_15'] = (((features['Predicted_x'] - features['Final_x'])*(features['Predicted_x'] - features['Final_x'])) + ((features['Predicted_y'] - features['Final_y'])*(features['Predicted_y'] - features['Final_y'])))
features['EPE_15'] = features['EPE_15'].apply(lambda x: math.sqrt(x))

features = features.reset_index()
del features['index']

train = features[0:int(len(features)*0.8)]
val = features[int(len(features)*0.8):]

print('Constant velocity val set EPE           :',round(val['EPE_15'].mean(),0))
print('Constant velocity val set MSE@15        :',round(val['MSE_15'].mean(),0))
print('Constant velocity val set MSE@10        :',round(val['MSE_10'].mean(),0))
print('Constant velocity val set MSE@5         :',round(val['MSE_5'].mean(),0))

assert len(train) + len(val) == len(features)
features.to_pickle(PATH + 'bdd10k_cv.pkl')

train = train.reset_index()
val = val.reset_index()
del train['index']
del val['index']

print('Saving...')
train.to_pickle(PATH + 'bdd10k_train.pkl')
val.to_pickle(PATH + 'bdd10k_val.pkl')

print('Done.')
