'''
This file processes bounding boxes from the BDD-10K dataset and stores them in
pickle format. The code is similar to processes_bounding_boxes_jaad.py but is
more efficient and adapted for BDD.

The processed boxes contain the following important fields:

Labeled:            Is this frame labeled? The pedestrian must have been tracked
                    for MIN_LENGTH_PAST previous frames and MIN_LENGTH_FUTURE
                    future frames.
Past_x, Past_y:     Past x and y coordinates of the bounding box centroid for
                    the previous MIN_LENGTH_PAST-1 frames and current frame
Future_x, Future_y: Future x and y coordinates of the bounding box centroid for
                    the future MIN_LENGTH_FUTURE frames
'''

import pandas as pd
import os
import cv2
from math import floor
import numpy as np

TRACK_PATH = '/home/olly/data/bdd/bdd_10k_detections.csv'
SAVE_PATH = '../data/'

# How far back and how far foward to use as features/predict respectively
MIN_LENGTH_PAST = 10
MIN_LENGTH_FUTURE = 15
MIN_DETECTION_LENGTH = MIN_LENGTH_PAST + MIN_LENGTH_FUTURE

boxes = pd.read_csv(TRACK_PATH)

boxes['Requires_features'] = 0
boxes['Labeled'] = 0

# BB format: top left (x,y), bottom right (x,y)
boxes['Mid_x'] = (boxes['bb1'] + boxes['bb3']) / 2
boxes['Mid_y'] = (boxes['bb2'] + boxes['bb4']) / 2
boxes['Height'] = boxes['bb4'] - boxes['bb2']
boxes['Width'] = boxes['bb3'] - boxes['bb1']

boxes['Past_x'] = 0
boxes['Past_y'] = 0
boxes['Future_x'] = 0
boxes['Future_y'] = 0
boxes['Past_x'] = boxes['Past_x'].astype(object)
boxes['Past_y'] = boxes['Past_y'].astype(object)
boxes['Future_x'] = boxes['Future_x'].astype(object)
boxes['Future_y'] = boxes['Future_y'].astype(object)

# Remove small pedestrians
boxes = boxes[boxes['Height']>50]
boxes = boxes.sort_values(by=['filename','track','frame_num'])
boxes = boxes.reset_index()
del boxes['index']

# Flag labeled images
boxes['Labeled'] = np.where(boxes['detection_length'] >= MIN_DETECTION_LENGTH,1,0)
boxes['Labeled'] = boxes['Labeled'].shift(-MIN_LENGTH_FUTURE)

print('Storing centroids. This make take a few minutes.')

# Store MIN_LENGTH_PAST and future MIN_LENGTH_FUTURE bounding box centroids
past_x_names = []
for past in range(MIN_LENGTH_PAST,0,-1):
    boxes['prev_x' + str(past)] = boxes['Mid_x'].shift(past)
    past_x_names.append('prev_x' + str(past))
past_y_names = []
for past in range(MIN_LENGTH_PAST,0,-1):
    boxes['prev_y' + str(past)] = boxes['Mid_y'].shift(past)
    past_y_names.append('prev_y' + str(past))
future_x_names = []
for future in range(1,MIN_LENGTH_FUTURE+1):
    boxes['future_x' + str(future)] = boxes['Mid_x'].shift(-future)
    future_x_names.append('future_x' + str(future))
future_y_names = []
for future in range(1,MIN_LENGTH_FUTURE+1):
    boxes['future_y' + str(future)] = boxes['Mid_y'].shift(-future)
    future_y_names.append('future_y' + str(future))

boxes['Past_x'] = boxes[past_x_names].values.tolist()
boxes['Past_y'] = boxes[past_y_names].values.tolist()
boxes['Future_x'] = boxes[future_x_names].values.tolist()
boxes['Future_y'] = boxes[future_y_names].values.tolist()

boxes = boxes.dropna(subset=['filename'],axis=0)

boxes.to_pickle(SAVE_PATH + 'bdd_10k_location_features.pkl')
print('Done.')
