'''
This file takes a set of detections and tracks represented by a CSV file and
counts the length of each track.
CSV sould be of the format:
filename,frame_num,bb1,bb2,bb3,bb4,track
where (bb1,bb2) = (x,y) top left and (bb1,bb2) = (x,y) bottom right
Returns a CSV with counted detection lengths.

NOTE: This code is not optimized and may take a long time to run. It also
assumes that the maximum number of detections in a single frame is 100.
'''
import pandas as pd
import cv2
import sys
try:
    detections = pd.read_csv(sys.argv[1])
except:
    print('Failed to load detections file.')
    sys.exit('Usage: python find_detectoion_length.py PATH_TO_DETECTIONS/file.csv')

detections['detection_length'] = 0
detections['Height'] = detections['bb4'] - detections['bb2']
# Remove small pedestrians
detections = detections[detections['Height']>50]
# Remove duplicates
detections = detections.drop_duplicates(subset = ['filename','track','frame_num'],keep=False)
detections = detections.sort_values(by=['filename','track','frame_num'])
detections = detections.reset_index()
del detections['index']
print('Total size: ', len(detections))
for ix,row in detections.iterrows():
    if ix % 100 == 0:
        print(ix,' of ',len(detections))
    # Remove rows that we know we do not need to search through for speedup
    # 100 is assumed to be the max number of pedestrians detected in a single frame
    if ix > 100:
        nearby_detections = detections[ix-100:ix]
    else:
        nearby_detections = detections

    if len(nearby_detections[(nearby_detections['filename'] == row['filename']) & (nearby_detections['frame_num'] == row['frame_num']-1) & (nearby_detections['track'] == row['track'])]) != 0:
        detections.loc[ix,'detection_length'] = nearby_detections[(nearby_detections['filename'] == row['filename']) & (nearby_detections['frame_num'] == row['frame_num']-1) & (nearby_detections['track'] == row['track'])]['detection_length'].values + 1
detections.to_csv('./detections_with_track_length.csv',index=False)
