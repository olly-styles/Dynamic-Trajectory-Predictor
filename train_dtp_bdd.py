import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import copy
import argparse

import pandas as pd
import numpy as np
import math
from PIL import Image
import video_transforms
from scipy.misc import imresize
import scipy.ndimage
import model_utils as utils
import sys

def get_modified_resnet(NUM_FLOW_FRAMES):
    '''
    Returns a ResNet18 model with the first layer of shape NUM_FLOW_FRAMES*2
    and output layer of shape 30.
    Applys partial batch norm and cross-modalitity pre-training following
    TSN:  https://arxiv.org/abs/1608.00859
    '''
    model = models.resnet18(pretrained=True)
    # Reshape resnet
    model = model.apply(utils.freeze_bn)
    model.bn1.train(True)

    pretrained_weights = model.conv1.weight
    avg_weights = torch.mean(pretrained_weights, 1)
    avg_weights = avg_weights.expand(NUM_FLOW_FRAMES*2,-1,-1,-1)
    avg_weights = avg_weights.permute(1,0,2,3)
    model.conv1 = nn.Conv2d(NUM_FLOW_FRAMES*2, 64, kernel_size=7, stride=2, padding=3)
    model.conv1.weight.data = avg_weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 30)

    return model

class DynamicTrajectoryPredictor(nn.Module):
    '''
    Creates the DTP model
    '''
    def __init__(self,NUM_FLOW_FRAMES):
        super(DynamicTrajectoryPredictor, self).__init__()

        # Encoder flow
        self.resnet18 = get_modified_resnet(NUM_FLOW_FRAMES)

    def forward(self, flow):

        out = self.resnet18(flow)

        return out


class LocationDatasetBDD(Dataset):
    def __init__(self, filename, root_dir, img_root, transform,NUM_FLOW_FRAMES,proportion=100):
        """
        Args:
            filename (string): Pkl file name with data. This must contain the
            optical flow image filename and the label.
            root_dir (string): Path to directory with the pkl file.
            proportion(int): Proportion of dataset to use for training
                            (up to 100, which is 100 percent of the dataset)
        """
        np.random.seed(seed=26) # Set seed for reproducability
        self.df = pd.read_pickle(root_dir + filename)
        print('Loaded data from ',root_dir + filename)
        unique_filenames = self.df['filename'].unique()
        np.random.shuffle(unique_filenames)
        unique_filenames = unique_filenames[0:int(len(unique_filenames) * proportion/100)]
        self.df = self.df[self.df['filename'].isin(unique_filenames)]
        self.df = self.df.reset_index()
        self.transform = transform
        self.img_root = img_root
        self.NUM_FLOW_FRAMES = NUM_FLOW_FRAMES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Returns:
            sample (dict): Containing:
                flow_stack (np.array):  Stack of optical flow images of shape
                                        256,256,NUM_FLOW_FRAMES*2
                label:                  Label of format [x,x,x...y,y,y...]
        '''
        # Labels are the CV correction term
        label_x = self.df.loc[idx, 'E_x']
        label_y = self.df.loc[idx, 'E_y']
        NUM_FLOW_FRAMES = self.NUM_FLOW_FRAMES

        dir_name = self.df.loc[idx, 'filename']
        track = self.df.loc[idx, 'track']

        label = np.array([label_x,label_y])
        label = label.flatten()

        # Frame number is part of the filename
        frame_num = self.df.loc[idx, 'frame_num']


        flow_stack = np.zeros((256,256,NUM_FLOW_FRAMES*2)).astype('uint8')

        # Read in the optical flow images
        for frame in range(frame_num+1-NUM_FLOW_FRAMES,frame_num+1):
            frame_name = dir_name + '/frame_' + str(frame).zfill(4) + '_ped_' + str(int(track)) + '.png'
            img_name_hor = str(self.img_root + 'bdd-horizontal/' + \
                                    frame_name)
            img_name_ver = str(self.img_root + 'bdd-vertical/' + \
                                    frame_name)

            try:
                hor_flow = Image.open(img_name_hor).resize((256,256))
                ver_flow = Image.open(img_name_ver).resize((256,256))
            except:
                print('Error: file not loaded. Could not find image file:')
                print(img_name_hor)
                hor_flow = np.zeros((256,256))
                ver_flow = np.zeros((256,256))

            flow_stack[:,:,int((frame-frame_num-1+NUM_FLOW_FRAMES)*2)] = hor_flow
            flow_stack[:,:,int(((frame-frame_num-1+NUM_FLOW_FRAMES)*2)+1)] = ver_flow

        flow_stack=self.transform(flow_stack)

        sample = {'flow_stack':flow_stack, 'labels': label}
        return sample


def train(model, device, train_loader, optimizer, epoch,loss_function):
    '''
    Trains DTP
    args:
        model: DTP as defined in the DynamicTrajectoryPredictor class
        device: GPU or CPU
        train_loader: Dataloader to produce stacks of optical flow images
        optimizer: eg. ADAM
        epoch: Current epoch (for printing progress)
        loss_function: eg. MSE
    '''
    model.train()
    all_outputs_5 = np.array([])
    all_targets_5 = np.array([])
    all_outputs_10 = np.array([])
    all_targets_10 = np.array([])
    all_outputs_15 = np.array([])
    all_targets_15 = np.array([])
    for batch_idx, data in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print('Batch ',batch_idx,' of ', len(train_loader))
        if batch_idx > 3000:
            break

        flow, targets = data['flow_stack'].to(device), data['labels'].to(device)

        targets = targets.float()

        optimizer.zero_grad()

        flow = flow.float()

        output = model(flow)

        loss = loss_function(output, targets)

        loss.backward()
        optimizer.step()

        output_5 = torch.cat((output[:,0:5],output[:,15:20]),dim=1)
        output_10 = torch.cat((output[:,0:10],output[:,15:25]),dim=1)
        output_15 = output

        targets_5 = torch.cat((targets[:,0:5],targets[:,15:20]),dim=1)
        targets_10 = torch.cat((targets[:,0:10],targets[:,15:25]),dim=1)
        targets_15 = targets

        all_outputs_5 = np.append(all_outputs_5,output_5.detach().cpu().numpy())
        all_targets_5 = np.append(all_targets_5,targets_5.detach().cpu().numpy())
        all_outputs_10 = np.append(all_outputs_10,output_10.detach().cpu().numpy())
        all_targets_10 = np.append(all_targets_10,targets_10.detach().cpu().numpy())
        all_outputs_15 = np.append(all_outputs_15,output_15.detach().cpu().numpy())
        all_targets_15 = np.append(all_targets_15,targets_15.detach().cpu().numpy())

    MSE_5 = utils.calc_mse(all_outputs_5,all_targets_5)
    FDE_5 = utils.calc_fde(all_outputs_5,all_targets_5,n=5)
    MSE_10 = utils.calc_mse(all_outputs_10,all_targets_10)
    FDE_10 = utils.calc_fde(all_outputs_10,all_targets_10,n=10)
    MSE_15 = utils.calc_mse(all_outputs_15,all_targets_15)
    FDE_15 = utils.calc_fde(all_outputs_15,all_targets_15,n=15)

    print('Train Epoch: {} [{}/{} ({:.0f}%)] \tMSE@5: {:.0f} \tFDE@5: {:.0f} \tMSE@10: {:.0f} \tFDE@10: {:.0f} \tMSE@15: {:.0f} \tFDE@15: {:.0f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), MSE_5, FDE_5,MSE_10, FDE_10,MSE_15, FDE_15))

def test(model, device, test_loader, loss_function):
    '''
    Evaluates DTP
    args:
        model: DTP as defined in the DynamicTrajectoryPredictor class
        device: GPU or CPU
        test_loader: Dataloader to produce stacks of optical flow images
        loss_function: eg. MSE
    returns:
        MSE and FDE at intervals of 5,10,15 frames into the future
        Outputs and targets 15 frames into the future
    '''
    model.eval()
    test_loss = 0
    all_outputs_5 = np.array([])
    all_targets_5 = np.array([])
    all_outputs_10 = np.array([])
    all_targets_10 = np.array([])
    all_outputs_15 = np.array([])
    all_targets_15 = np.array([])

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print('Batch ',batch_idx,' of ', len(test_loader))

            flow, targets = data['flow_stack'].to(device), data['labels'].to(device)
            flow = flow.float()
            targets = targets.float()

            output = model(flow)

            test_loss += loss_function(output, targets).item() # sum up batch loss


            output_5 = torch.cat((output[:,0:5],output[:,15:20]),dim=1)
            output_10 = torch.cat((output[:,0:10],output[:,15:25]),dim=1)
            output_15 = output

            targets_5 = torch.cat((targets[:,0:5],targets[:,15:20]),dim=1)
            targets_10 = torch.cat((targets[:,0:10],targets[:,15:25]),dim=1)
            targets_15 = targets

            all_outputs_5 = np.append(all_outputs_5,output_5.detach().cpu().numpy())
            all_targets_5 = np.append(all_targets_5,targets_5.detach().cpu().numpy())
            all_outputs_10 = np.append(all_outputs_10,output_10.detach().cpu().numpy())
            all_targets_10 = np.append(all_targets_10,targets_10.detach().cpu().numpy())
            all_outputs_15 = np.append(all_outputs_15,output_15.detach().cpu().numpy())
            all_targets_15 = np.append(all_targets_15,targets_15.detach().cpu().numpy())

    MSE_5 = utils.calc_mse(all_outputs_5,all_targets_5)
    FDE_5 = utils.calc_fde(all_outputs_5,all_targets_5,n=5)
    MSE_10 = utils.calc_mse(all_outputs_10,all_targets_10)
    FDE_10 = utils.calc_fde(all_outputs_10,all_targets_10,n=10)
    MSE_15 = utils.calc_mse(all_outputs_15,all_targets_15)
    FDE_15 = utils.calc_fde(all_outputs_15,all_targets_15,n=15)

    print('Validation: \t\t\t\tMSE@5: {:.0f} \tFDE@5: {:.0f} \tMSE@10: {:.0f} \tFDE@10: {:.0f} \tMSE@15: {:.0f} \tFDE@15: {:.0f}'.format(
         MSE_5, FDE_5,MSE_10, FDE_10,MSE_15, FDE_15))
    return MSE_5,FDE_5,MSE_10,FDE_10,MSE_15,FDE_15, all_outputs_15, all_targets_15

def main(args):
    ############################################################################
    # Path to optical flow images
    if args.detector == 'yolo':
        img_root = './data/yolov3/'
    else:
        img_root = './data/faster-rcnn/'
    # Path to training and testing files
    load_path = './data/'
    # CPU or GPU?
    device = torch.device("cuda")

    # Model saving and loading
    model_save_path = './data/'
    model_load_path = './data/'

    # Training settings
    epochs = 15
    batch_size = 64
    learning_rate = 1e-5
    num_workers = 8
    weight_decay = 1e-2
    NUM_FLOW_FRAMES = 9
    training_proportion = 100 # How much of the dataset to use? 100 = 100percent

    # Transformers for training and validation
    transform_train = video_transforms.Compose([
            video_transforms.MultiScaleCrop((224, 224), [1.0]),
            video_transforms.ToTensor(),
        ])

    transform_val = video_transforms.Compose([
            video_transforms.Scale((224)),
            video_transforms.ToTensor(),
        ])

    ############################################################################

    print('################### Training settings ###################')
    print(  'epochs:', epochs,
            '   batch_size:', batch_size,
            '   learning_rate:', learning_rate,
            '   num_workers:', num_workers,
            '   NUM_FLOW_FRAMES:', NUM_FLOW_FRAMES)

    results = pd.DataFrame()

    print('Training model')
    print(args.detector + '_bdd10k_val.pkl')

    try:
        testset = LocationDatasetBDD(filename='bdd10k_val_' + args.detector + '.pkl',
                                    root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        trainset = LocationDatasetBDD(filename='bdd10k_train_' + args.detector + '.pkl',
                                    root_dir=load_path, transform=transform_train, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES, proportion=training_proportion)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
        valset = LocationDatasetBDD(filename='bdd10k_val_' + args.detector + '.pkl',
                                    root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)
    except:
        sys.exit('ERROR: Could not load pkl data file. Check the bdd .pkl files are in the correct path.')

    model = DynamicTrajectoryPredictor(NUM_FLOW_FRAMES).to(device)
    model = model.float()

    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.MSELoss()
    best_FDE = np.inf
    best_MSE = np.inf
    best_model = copy.deepcopy(model)

    # Begin training
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_function)
        MSE_5,FDE_5,MSE_10,FDE_10,MSE_15,FDE_15,_,_ = test(model, device, val_loader, loss_function)
        if MSE_15 < best_MSE:
            best_MSE = MSE_15
            best_model = copy.deepcopy(model)
            best_FDE = FDE_15
            torch.save(best_model.state_dict(), model_save_path + args.detector + '_rn18_bdd10k_flow_css_'+ str(NUM_FLOW_FRAMES) + 'stack_training_proportion_' + str(training_proportion) +  '_shuffled_disp.weights')
        print(epoch)
        print('Best MSE:',round(best_MSE,0))

    test_mse_5,test_fde_5,test_mse_10,test_fde_10,test_mse_15,test_fde_15,all_outputs,all_targets = test(best_model, device, test_loader, loss_function)
    print('Test mse @ 15:', round(test_mse_15,0))

    # Save the model
    torch.save(best_model.state_dict(), model_save_path + args.detector + 'bdd10k_rn18_flow_css_'+ str(NUM_FLOW_FRAMES) + 'stack_training_proportion_' + str(training_proportion) + '_shuffled_disp.weights')

    # Save the predictions and the targets
    np.save('./' + args.detector + '_predictions_rn18_flow_css_'+ str(NUM_FLOW_FRAMES) + 'stack_bdd10k_training_proportion_' + str(training_proportion) +'_shuffled_disp.npy',all_outputs)
    np.save('./' + args.detector + '_targets_rn18_flow_css_' + str(NUM_FLOW_FRAMES) + 'stack_bdd10k__shuffled_disp.npy',all_targets)

    # Save the results
    result = {'NUM_FLOW_FRAMES':NUM_FLOW_FRAMES,'training_proportion':training_proportion,'val_mse':best_MSE,'val_fde':best_FDE, 'test_mse_5' :test_mse_5,'test_fde_5' :test_fde_5,'test_mse_10' :test_mse_10,'test_fde_10' :test_fde_10,'test_mse_15' :test_mse_15,'test_fde_15' :test_fde_15}
    results = results.append(result, ignore_index=True)
    results.to_csv('./' + args.detector + '_results_rn18_bdd10k.csv',index=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', '-d', help="Use detections from 'yolo' or 'faster-rcnn'", type= str, default='yolo')
    args = parser.parse_args()
    main(args)
