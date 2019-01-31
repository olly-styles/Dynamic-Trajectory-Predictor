import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import copy
import sys
import argparse

import pandas as pd
import numpy as np
import math
from PIL import Image
import video_transforms
from scipy.misc import imresize
import scipy.ndimage
import model_utils as utils

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

class LocationDatasetJAAD(Dataset):
    def __init__(self, filename, root_dir, img_root, transform,NUM_FLOW_FRAMES):
        """
        Args:
            filename (string): Pkl file name with data. This must contain the
            optical flow image filename and the label.
            root_dir (string): Path to directory with the pkl file.
        """
        self.df = pd.read_pickle(root_dir + filename)
        print('Loaded data from ',root_dir + filename)
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
                filename
        '''
        # Labels are the CV correction term
        label_x = self.df.loc[idx, 'E_x']
        label_y = self.df.loc[idx, 'E_y']
        NUM_FLOW_FRAMES = self.NUM_FLOW_FRAMES
        filename = self.df.loc[idx, 'Filename']

        label = np.array([label_x,label_y])
        label = label.flatten()

        # Frame number is part of the filename
        frame_num = int(filename.split('_')[2])

        flow_stack = np.zeros((256,256,NUM_FLOW_FRAMES*2)).astype('uint8')

        # Read in the optical flow images
        for frame in range(frame_num+1-NUM_FLOW_FRAMES,frame_num+1):
            frame_name = filename[0:17] + str(frame).zfill(4) + filename[21:]
            img_name_hor = str(self.img_root + 'jaad-horizontal/' + \
                                    frame_name)
            img_name_ver = str(self.img_root + 'jaad-vertical/' + \
                                    frame_name)
            try:
                img_name_hor = '.'.join(img_name_hor.split('.')[0:-1]) + '.jpg'
                img_name_ver = '.'.join(img_name_ver.split('.')[0:-1]) + '.jpg'

                hor_flow = Image.open(img_name_hor).resize((256,256))
                ver_flow = Image.open(img_name_ver).resize((256,256))
            except:
                print('Error: file not loaded. Could not find image file: ')
                print(img_name_hor)
                hor_flow = np.zeros((256,256))
                ver_flow = np.zeros((256,256))

            flow_stack[:,:,int((frame-frame_num-1+NUM_FLOW_FRAMES)*2)] = hor_flow
            flow_stack[:,:,int(((frame-frame_num-1+NUM_FLOW_FRAMES)*2)+1)] = ver_flow

        flow_stack=self.transform(flow_stack)

        sample = {'flow_stack':flow_stack, 'labels': label, 'filenames': filename}
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
    img_root = './data/human-annotated/'
    # Path to training and testing files
    load_path = './data/'
    # CPU or GPU?
    device = torch.device("cuda")

    # Training settings
    epochs = 30
    batch_size = 64
    learning_rate = 1e-5
    num_workers = 8
    pretrained = False
    weight_decay = 1e-2
    NUM_FLOW_FRAMES = 9

    model_load_path = args.model_load_path
    model_save_path = args.model_save_path

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
            '   model_load_path:', model_load_path,
            '   NUM_FLOW_FRAMES:', NUM_FLOW_FRAMES)

    results = pd.DataFrame()

    for fold in [1,2,3,4,5]:
        if pretrained:
            learning_rate = 1e-6
            epochs = 30
        else:
            learning_rate = 1e-5
            epochs = 40

        print('Training on fold ' + str(fold))

        try:
            testset = LocationDatasetJAAD(filename='jaad_cv_test.pkl',
                                        root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
            trainset = LocationDatasetJAAD(filename='jaad_cv_train_' + str(fold) + '.pkl',
                                        root_dir=load_path, transform=transform_train, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, num_workers=num_workers)
            valset = LocationDatasetJAAD(filename='jaad_cv_val_' + str(fold) + '.pkl',
                                        root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
        except:
            sys.exit('ERROR: Could not load pkl data file. Check the jaad .pkl files are in the correct path.')

        model = DynamicTrajectoryPredictor(NUM_FLOW_FRAMES).to(device)
        model = model.float()

        model = nn.DataParallel(model)

        if model_load_path is not None:
            print('loading model from', model_load_path)
            model.load_state_dict(torch.load(model_load_path))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_function = torch.nn.MSELoss()
        best_FDE = np.inf
        best_MSE = np.inf
        best_model = copy.deepcopy(model)

        # Begin training
        for epoch in range(1, epochs + 1):
            # Set learning rate to 1e-6 after 30 epochs
            if epoch > 30:
                optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_weight_decay=decay)

            train(model, device, train_loader, optimizer, epoch, loss_function)
            MSE_5,FDE_5,MSE_10,FDE_10,MSE_15,FDE_15,_,_ = test(model, device, val_loader, loss_function)
            if MSE_15 < best_MSE:
                best_MSE = MSE_15
                best_model = copy.deepcopy(model)
                best_FDE = FDE_15
            print(epoch)
            print('Best MSE:',round(best_MSE,0))

        test_mse_5,test_fde_5,test_mse_10,test_fde_10,test_mse_15,test_fde_15,all_outputs,all_targets = test(best_model, device, test_loader, loss_function)
        print('Test mse @ 15:', round(test_mse_15,0))

        # Save the model
        torch.save(best_model.state_dict(), model_save_path + 'rn18_flow_css_'+ str(NUM_FLOW_FRAMES) + 'stack_fold_' + str(fold) + '_pretrained-' + str(pretrained) +'_disp.weights')

        # Save the predictions and the targets
        np.save('./predictions_rn18_flow_css_'+ str(NUM_FLOW_FRAMES) + 'stack_jaad_fold_' + str(fold) + 'pretrained-' + str(pretrained) +'_disp.npy',all_outputs)
        np.save('./targets_rn18_flow_css_' + str(NUM_FLOW_FRAMES) + 'stack_jaad_fold_' + str(fold) + 'pretrained-' + str(pretrained) +'_disp.npy',all_targets)

        # Save the results
        result = {'NUM_FLOW_FRAMES':NUM_FLOW_FRAMES,'fold':fold, 'val_mse':best_MSE,'val_fde':best_FDE, 'test_mse_5' :test_mse_5,'test_fde_5' :test_fde_5,'test_mse_10' :test_mse_10,'test_fde_10' :test_fde_10,'test_mse_15' :test_mse_15,'test_fde_15' :test_fde_15,'pretrained':pretrained}
        results = results.append(result, ignore_index=True)
        results.to_csv('./results_rn18_jaad.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_load_path', '-l', help="Path to load model", type= str, default=None)
    parser.add_argument('--model_save_path', '-s', help="Path to save model", type= str, default= './')

    args = parser.parse_args()

    main(args)
