# Forecasting Pedestrian Trajectory from Unlabelled Training Data
This repository contains the code for the paper: Forecasting Pedestrian Trajectory from Unlabelled Training Data.

Paper: https://arxiv.org/pdf/1905.03681.pdf  
  
Video: https://www.youtube.com/watch?v=jUTQyUjeynE

## Installation
#### Clone Repo 
```bash
git clone https://github.com/olly-styles/Dynamic-Trajectory-Predictor
cd Dynamic-Trajectory-Predictor/
```
#### Create virtual environment (recommended)
```bash
virtualenv --no-site-packages dtp
source dtp/bin/activate
```
#### Install packages
```bash
pip install -r requirements.txt
```
## Training on JAAD
#### Download data
Requires data downloads from Google Drive. This can be done manually, or using gdown (below)
```bash
mkdir data && cd data
gdown https://drive.google.com/uc?id=1OuXLKrB6ItikYbnCM1yODQk2IUAmQ07y
gdown https://drive.google.com/uc?id=1mP4y-S8NEnavfGGZLCzkfw4EIpDUfJnp
unzip human-annotated.zip
```
#### Preprocess data
```bash
cd ../preprocessing/
python process_bounding_boxes_jaad.py
python compute_cv_correction_jaad.py
cd ..
```
#### Train from scratch
```bash
python train_dtp_jaad.py
```
#### Fine-tune the pre-trained model
```bash
cd data && mkdir models && cd models
gdown https://drive.google.com/uc?id=1J2VclWeEjMj7WQhTmEPhjCaza4w5PSmX
cd ..
python train_dtp_jaad.py -l ./data/models/bdd10k_rn18_flow_css_9stack_training_proportion_100_shuffled_disp.weights
```

## Running on BDD
#### Yolov3
```bash
cd data
gdown https://drive.google.com/uc?id=17Fvkrtxg_NEH2edH-wEp_Po5Y777zGQJ
gdown https://drive.google.com/uc?id=1mcL-c-FT19ePFdaLu8v1rmApoLDIeGYe
unzip yolov3.zip
python process_bounding_boxes_bdd.py
python compute_cv_correction_bdd.py
python train_dtp_bdd.py
```
#### Faster-RCNN
```bash
cd data
gdown https://drive.google.com/uc?id=1SNVe9SSRYiG-6WQZpvIOl_KtxfAThG2y
gdown https://drive.google.com/uc?id=1hKbnGThFS-shggFraQMuGhl9gy0E7ylV
unzip faster-rcnn.zip
cd ../preprocessing
python process_bounding_boxes_bdd.py -d faster-rcnn
python compute_cv_correction_bdd.py -d faster-rcnn
cd ..
python train_dtp_bdd.py -d faster-rcnn
```




