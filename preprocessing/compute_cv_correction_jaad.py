'''
Before running this file bounding boxes must be processed using
process_bounding_boxes_jaad.py

This file takes the processed bounding boxes and does the following:
    1. Computes the pedestrians velocity and prints the error under the assumption
    that the pedestrian will maintain their veloicty
    2. Computes E_x and E_y. This is the value to be predicted by the model using
    the constant veloicty correction term. C_x = 1 - E_x
    3. Splits the data training and testing
    4. Splits the training data into 5 folds
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
MIN_LENGTH_FUTURE = 15
VELOCITY_FRAMES = 5
VELOCITY_FRAMES=MIN_LENGTH_PAST-VELOCITY_FRAMES

features = pd.read_pickle(PATH + 'jaad_location_features.pkl')
features = features[features['Labeled']==1]

##################### Compute velocity and make predictions #####################
features['Final_x'] = features['Future_x'].apply(lambda x: x[-1])
features['Final_y'] = features['Future_y'].apply(lambda y: y[-1])

features['Velocity_x'] = features['Past_x'].apply(lambda x: utils.mean_velocity(x,VELOCITY_FRAMES))
features['Velocity_y'] = features['Past_y'].apply(lambda y: utils.mean_velocity(y,VELOCITY_FRAMES))

features['Predicted_x'] = features['Mid_x'] + (MIN_LENGTH_FUTURE * features['Velocity_x'])
features['Predicted_y'] = features['Mid_y'] + (MIN_LENGTH_FUTURE * features['Velocity_y'])

features['Predicted_x_seq'] = features.apply(lambda x: utils.get_seq_preds(x['Mid_x'],x['Velocity_x'],MIN_LENGTH_FUTURE),axis=1 )
features['Predicted_y_seq'] = features.apply(lambda x: utils.get_seq_preds(x['Mid_y'],x['Velocity_y'],MIN_LENGTH_FUTURE),axis=1 )

features['E_x'] = features.apply(lambda x: (x['Future_x']-x['Predicted_x_seq']),axis=1 )
features['E_y'] = features.apply(lambda y: (y['Future_y']-y['Predicted_y_seq']),axis=1 )

##################### Get errors #####################
features['MSE'] = features.apply(lambda x: utils.calc_mse(x['Predicted_x_seq'],x['Future_x'],x['Predicted_y_seq'],x['Future_y'],15),axis=1 )
features['MSE_10'] = features.apply(lambda x: utils.calc_mse(x['Predicted_x_seq'],x['Future_x'],x['Predicted_y_seq'],x['Future_y'],10),axis=1 )
features['MSE_5'] = features.apply(lambda x: utils.calc_mse(x['Predicted_x_seq'],x['Future_x'],x['Predicted_y_seq'],x['Future_y'],5),axis=1 )

features['EPE'] = (((features['Predicted_x'] - features['Final_x'])*(features['Predicted_x'] - features['Final_x'])) + ((features['Predicted_y'] - features['Final_y'])*(features['Predicted_y'] - features['Final_y'])))
features['EPE'] = features['EPE'].apply(lambda x: math.sqrt(x))

##################### Pre-process filenames #####################
features['Video'] = features['Video'].apply(lambda x: x.split('.')[0])
features['Video'] = features['Video'].apply(lambda x: x[0:5] + '_' + x[5:])

features['Frame'] = features['Frame'].astype(int).astype(str).apply(lambda x: x.zfill(4))
features['Filename'] = features['Video'] + '/frame_' + features['Frame'] + '_ped_' + features['Pedestrian'].astype(str) + '.png'

features = features.reset_index()
del features['index']
features.to_pickle(PATH + 'jaad_cv.pkl')

##################### Split into train and test. Compute error #####################
train = features[features['Video'] < 'video_0250.mp4']
test = features[features['Video'] >= 'video_0250.mp4']

test = test.reset_index()
del test['index']

print('Constant velocity test set EPE               :',round(test['EPE'].mean(),1))
print('Constant velocity test set MSE@15            :',round(test['MSE'].mean(),0))
print('Constant velocity test set MSE@10            :',round(test['MSE_10'].mean(),0))
print('Constant velocity test set MSE@5             :',round(test['MSE_5'].mean(),0))


##################### Split into 5 folds #####################

# 200 - 250
train_1 = train[train['Video'] < 'video_0200.mp4']
val_1 = train[train['Video'] >= 'video_0200.mp4']
assert len(train_1) + len(val_1) == len(train)
assert(len(set(train_1.Video.unique()).intersection(set(val_1.Video.unique()))) == 0)

# 150 - 200
train_2 = train[(train['Video']<='video_0150.mp4') | (train['Video']>'video_0200.mp4')]
val_2 = train[(train['Video']>'video_0150.mp4') & (train['Video']<='video_0200.mp4')]
assert len(train_2) + len(val_2) == len(train)
assert(len(set(train_2.Video.unique()).intersection(set(val_2.Video.unique()))) == 0)

# 100 - 150
train_3 = train[(train['Video']<='video_0100.mp4') | (train['Video']>'video_0150.mp4')]
val_3 = train[(train['Video']>'video_0100.mp4') & (train['Video']<='video_0150.mp4')]
assert len(train_3) + len(val_3) == len(train)
assert(len(set(train_3.Video.unique()).intersection(set(val_3.Video.unique()))) == 0)

# 50 - 100
train_4 = train[(train['Video']<='video_0050.mp4') | (train['Video']>'video_0100.mp4')]
val_4 = train[(train['Video']>'video_0050.mp4') & (train['Video']<='video_0100.mp4')]
assert len(train_4) + len(val_4) == len(train)
assert(len(set(train_4.Video.unique()).intersection(set(val_4.Video.unique()))) == 0)

# 0 - 50
train_5 = train[train['Video']>'video_0050.mp4']
val_5 = train[train['Video']<='video_0050.mp4']
assert len(train_5) + len(val_5) == len(train)
assert(len(set(train_5.Video.unique()).intersection(set(val_5.Video.unique()))) == 0)

train_1 = train_1.reset_index()
val_1 = val_1.reset_index()
del train_1['index']
del val_1['index']

train_2 = train_2.reset_index()
val_2 = val_2.reset_index()
del train_2['index']
del val_2['index']

train_3 = train_3.reset_index()
val_3 = val_3.reset_index()
del train_3['index']
del val_3['index']

train_4 = train_4.reset_index()
val_4 = val_4.reset_index()
del train_4['index']
del val_4['index']

train_5 = train_5.reset_index()
val_5 = val_5.reset_index()
del train_5['index']
del val_5['index']

train_1.to_pickle(PATH + 'jaad_cv_train_1.pkl')
val_1.to_pickle(PATH + 'jaad_cv_val_1.pkl')
train_2.to_pickle(PATH + 'jaad_cv_train_2.pkl')
val_2.to_pickle(PATH + 'jaad_cv_val_2.pkl')
train_3.to_pickle(PATH + 'jaad_cv_train_3.pkl')
val_3.to_pickle(PATH + 'jaad_cv_val_3.pkl')
train_4.to_pickle(PATH + 'jaad_cv_train_4.pkl')
val_4.to_pickle(PATH + 'jaad_cv_val_4.pkl')
train_5.to_pickle(PATH + 'jaad_cv_train_5.pkl')
val_5.to_pickle(PATH + 'jaad_cv_val_5.pkl')
test.to_pickle(PATH + 'jaad_cv_test.pkl')
