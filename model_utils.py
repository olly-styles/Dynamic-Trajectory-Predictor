import numpy as np

def calc_fde(outputs,targets,n):
    '''
    Calculates the final displacement error (L2 distance) between outputs and
    targets (final output and final target)
    Args:
        outputs: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        targets: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        n: Number of predictions
    Returns:
        Final displacement error at n timesteps between outputs and targets
    '''

    # Reshape to [[x,y],[x,y],...)
    outputs = outputs.reshape(-1,n*2,order='A')
    outputs = outputs.reshape(-1,2,n)

    # Reshape to [[x,y],[x,y],...)
    targets = targets.reshape(-1,n*2,order='A')
    targets = targets.reshape(-1,2,n)

    # Get the final prediction
    outputs = outputs[:,:,-1]
    targets = targets[:,:,-1]

    # L2 Distance
    diff = (outputs - targets) * (outputs - targets)
    return np.mean(np.sqrt(np.sum(diff,axis=1)))

def calc_mse(outputs,targets):
    '''
    Calculates the mean squared error
    Args:
        outputs: np array. 1D array formated (x,x,x,x... y,y,y,y... )
        targets: np array. 1D array formated (x,x,x,x... y,y,y,y... )
    Returns:
        Mean squared error between outputs and targets
    '''

    diff = (outputs - targets) * (outputs - targets)

    return np.mean(diff)*2

def freeze_bn(m):
    '''
    Freezes the batch normalization of a layer.
    Taken from https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385
    '''
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()
