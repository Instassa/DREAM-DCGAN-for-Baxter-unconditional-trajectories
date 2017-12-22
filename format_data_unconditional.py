
"""
Created on Sat Dec 16 19:38:51 2017

@author: marija
"""
import numpy as np
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



mpl.rcParams['legend.fontsize'] = 10

with open('./data/archive_3600_trj.dat','r') as f:
    myarr = [L.strip().split(' ') for L in f]

Number_of_trajectories = np.shape(myarr)[0]
NumberT = 1000
arm = np.zeros(Number_of_trajectories); obj = np.zeros(np.shape(myarr)[0])
MinMax = np.zeros((Number_of_trajectories,2))
for i in range(Number_of_trajectories):
    traj  = np.array(myarr[i], dtype = float)
    arm[i] = traj[1]
    obj[i] = traj[2]
arm_l = np.array(arm, dtype = int)
obj_l = np.array(obj, dtype = int)
    
arm_l_max = arm_l.max(); print('Tn max: ', arm_l_max)
obj_l_max = obj_l.max(); print('To max: ', obj_l_max)

for i in range(NumberT):
    Tn_m = (3*np.int(traj[1])+4+1)
    MinMax[i, :] = [np.min(traj[4:Tn_m]), np.max(traj[4:Tn_m])]

LEN = arm_l_max 
LEN2 = 576


def converter(List):
    length = len(List)
    new_List = np.zeros(26)
    k = 0
    for i in range(length):
        try:           
            new_List[k] = float(List[i])
            k +=1
        except:
            k +=0
    new_List = new_List.reshape(26, 1)
    return(new_List)        

MAX = 11.904; 

for i in range(0,NumberT):
    fu  = './data/motion/{:0d}.dat'.format(i)
    with open(fu,'r') as f:
        myarr2 = [L.strip().split(' ') for L in f]
    
    traj_len = np.shape(myarr2)[0]
    gripper = 1; u = 0
    for j in range(traj_len):
        if j == 0:
            full_arm_traj = converter(myarr2[j])
        if gripper == 1:
            full_arm_traj = np.concatenate((full_arm_traj, converter(myarr2[j])), axis = 1)
            gripper = converter(myarr2[j])[22, 0]
            u +=1

    length = np.shape(full_arm_traj)[1]; interp_traj = np.zeros((np.shape(full_arm_traj)[0], arm_l_max))
    real_len = np.linspace(0, arm_l_max, length); intended_len = np.linspace(0, arm_l_max, arm_l_max)    

    for k in range(np.shape(full_arm_traj)[0]):
        interp_traj[k, :] = interpolate.pchip_interpolate(real_len, full_arm_traj[k, :], intended_len);
    RI = interp_traj.reshape((np.shape(interp_traj)[0], np.shape(interp_traj)[1], 1))/MAX; #[:, None]
    
    RI = RI[:, 0:(LEN):10, :]
    
    newpath = './data/formated_unconditional_trajectories/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
            
    f = './data/formated_unconditional_trajectories/npy_train_{:05d}.npy'.format(i)
    np.save(f, RI)
