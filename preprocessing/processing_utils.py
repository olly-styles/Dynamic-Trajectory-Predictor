import numpy as np

def mean_velocity(x,velocity_frames):
    #print(x)
    velocities = []
    for n in range(velocity_frames,len(x)):
        velocities.append(x[n]-x[n-1])
    return np.mean(velocities)

def get_seq_preds(curr_x,velocity,future):
    seq = np.arange(1,future+1)
    seq =  curr_x + (seq * velocity)
    return seq

def calc_mse(outputs_x,targets_x,outputs_y,targets_y,n):

    outputs_x = np.array(outputs_x)[0:n]
    targets_x = np.array(targets_x)[0:n]
    outputs_y = np.array(outputs_y)[0:n]
    targets_y = np.array(targets_y)[0:n]


    mse_x = np.mean((outputs_x - targets_x) * (outputs_x - targets_x))
    mse_y = np.mean((outputs_y - targets_y) * (outputs_y - targets_y))
    mse = mse_x + mse_y
    return mse
