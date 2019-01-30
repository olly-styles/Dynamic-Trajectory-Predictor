'''
This file processes bounding boxes from the JAAD dataset and stores them in
pickle format. It will adjust the bounding box coordinates from the original
resolution (a mix of 1920x1080 and 1280x720) to a consistent resolution of
1280x720. It will also adjust the sampling rate of the bounding boxes from a
mix of 30 and 60 FPS to a consistent rate of 15 FPS.

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
import numpy as np

TRACK_PATH = '../data/jaad_boxes.csv'

# Some videos are different frame rates and resolutions
SIXTY_FPS_VIDEOS = ['video0028.mp4','video0037.mp4','video0038.mp4','video0039.mp4','video0040.mp4','video0041.mp4','video0052.mp4','video0053.mp4']
LOW_RES_VIDEOS = ['video0061.mp4','video0062.mp4','video0063.mp4','video0064.mp4','video0065.mp4','video0066.mp4','video0067.mp4','video0068.mp4','video0069.mp4','video0070.mp4']
SAVE_PATH = '../data/'

# How far back and how far foward to use as features/predict respectively
MIN_LENGTH_PAST = 10
MIN_LENGTH_FUTURE = 15
MIN_DETECTION_LENGTH = MIN_LENGTH_PAST + MIN_LENGTH_FUTURE

boxes = pd.read_csv(TRACK_PATH)

# Remove bounding boxes that contain occlusion
boxes = boxes[boxes['Occlusion']==0]

boxes = boxes.sort_values(by=['Video','Pedestrian','Frame'])
boxes = boxes.drop_duplicates(subset = ['Video','Pedestrian','Frame'],keep=False)

# 0 index boxes
boxes['Frame'] = boxes['Frame'] - 1

# Downsample to 15 FPS
boxes = boxes[boxes['Frame']%2 == 0]
boxes = boxes[(boxes['Frame']%4 == 0) | (~boxes['Video'].isin(SIXTY_FPS_VIDEOS))]
boxes['Frame'] /= 2
boxes.loc[boxes['Video'].isin(SIXTY_FPS_VIDEOS),'Frame'] /= 2


boxes['Labeled'] = 0
boxes['Past_x'] = 0
boxes['Past_y'] = 0
boxes['Future_x'] = 0
boxes['Future_y'] = 0

boxes['Past_x'] = boxes['Past_x'].astype(object)
boxes['Past_y'] = boxes['Past_y'].astype(object)
boxes['Future_x'] = boxes['Future_x'].astype(object)
boxes['Future_y'] = boxes['Future_y'].astype(object)

boxes['detection_length'] = 0
# Some frames have been removed. Make a new index and delete the old one
boxes = boxes.reset_index()
del boxes['index']


#Adjust resolution
boxes.loc[~boxes['Video'].isin(LOW_RES_VIDEOS),'Mid_x'] *= (1280/1920)
boxes.loc[~boxes['Video'].isin(LOW_RES_VIDEOS),'Mid_y'] *= (720/1080)
boxes.loc[~boxes['Video'].isin(LOW_RES_VIDEOS),'Width'] *= (1280/1920)
boxes.loc[~boxes['Video'].isin(LOW_RES_VIDEOS),'Height'] *= (720/1080)
boxes.loc[~boxes['Video'].isin(LOW_RES_VIDEOS),'Start_x'] *= (1280/1920)
boxes.loc[~boxes['Video'].isin(LOW_RES_VIDEOS),'Start_y'] *= (720/1080)

# Remove small pedestrians
boxes = boxes[boxes['Height']>50]

boxes = boxes.sort_values(by=['Video','Pedestrian','Frame'])
boxes = boxes.reset_index()
del boxes['index']

# Label detection length, flag labeled images, and store the past MIN_LENGTH_PAST
# and future MIN_LENGTH_FUTURE bounding box centroids.
for rownum in range(1,len(boxes[1:])):
    if rownum % 100 == 0:
        print(rownum, ' of ', len(boxes[1:]))
    # Concecutive frame, same pedestrian, same video number
    if int(boxes.iloc[rownum]['Frame']) == int(boxes.iloc[rownum-1]['Frame'])+1 and \
        int(boxes.iloc[rownum]['Pedestrian']) == int(boxes.iloc[rownum-1]['Pedestrian']) and \
        boxes.iloc[rownum]['Video'] == boxes.iloc[rownum-1]['Video']:
        boxes.loc[rownum,'detection_length'] = boxes.iloc[rownum-1]['detection_length']+1
    else:
        boxes.loc[rownum,'detection_length'] = 0

    if boxes.iloc[rownum]['detection_length'] >= MIN_DETECTION_LENGTH:
        boxes.loc[rownum-MIN_LENGTH_FUTURE,'Labeled'] = 1
        past_x = []
        past_y = []

        future_x = []
        future_y = []

        past_x = (boxes.iloc[rownum-MIN_DETECTION_LENGTH+1:rownum-MIN_DETECTION_LENGTH+MIN_LENGTH_PAST+1]['Mid_x']).values
        past_y = (boxes.iloc[rownum-MIN_DETECTION_LENGTH+1:rownum-MIN_DETECTION_LENGTH+MIN_LENGTH_PAST+1]['Mid_y']).values

        future_x = (boxes.iloc[rownum-MIN_DETECTION_LENGTH+MIN_LENGTH_PAST+1:rownum+1]['Mid_x']).values
        future_y = (boxes.iloc[rownum-MIN_DETECTION_LENGTH+MIN_LENGTH_PAST+1:rownum+1]['Mid_y']).values


        boxes.at[rownum-MIN_LENGTH_FUTURE,'Past_x'] = past_x
        boxes.at[rownum-MIN_LENGTH_FUTURE,'Past_y'] = past_y
        boxes.at[rownum-MIN_LENGTH_FUTURE,'Future_x'] = future_x
        boxes.at[rownum-MIN_LENGTH_FUTURE,'Future_y'] = future_y

# Save the result as a pkl. Must be a pkl (rather than csv) to preserve data structure
boxes.to_pickle(SAVE_PATH + 'jaad_location_features.pkl')
