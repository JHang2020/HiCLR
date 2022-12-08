import torch
import torch.nn as nn
import numpy as np
import math
import random
#temoral-frame mask
class TemporalMask:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
        
    def __call__(self, data, frame):
        '''
        given a data: N,C,T,V,M
                frame: the number of the valid frames in data 
        return Mask: N,C,T,V,M 
        randomly mask the frame*ratio frames
        '''
        #data N,C,T,V,M
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        #data = data.reshape(N,T,C*V*M)
        size, max_frame, feature_dim = N,T,V*C*M
        #data = data.view(size, max_frame, self.person_num, self.joint_num, self.channel_num)#N,T,M,V,C

        mask_idx = torch.tensor((frame * (1 - self.mask_ratio))).reshape((1, 1, 1, 1, 1)).repeat(size, max_frame, self.person_num, self.joint_num, self.channel_num)
        frame_idx = torch.arange(max_frame).reshape((1, max_frame, 1, 1, 1)).repeat(size, 1, self.person_num, self.joint_num, self.channel_num)
        mask = (frame_idx < mask_idx).float()
        randper = np.random.permutation(range(frame))
        rand = torch.arange(max_frame)
        rand[0:frame] = torch.from_numpy(randper)
        mask = mask[:,rand,...]
        #N,T,M,V,C -> N C T V M
        #mask = mask.permute((0,4,1,3,2))
        #trans_data = data * mask
        #trans_data = trans_data.view(size, max_frame, feature_dim)
        return mask #N,T,M,V,C
    
#random joint mask different for frames
class Jointmask:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
    def __call__(self, data, frame):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((T,V))
        mask_joint_num = int(V*self.mask_ratio)
        for i in range(frame):
            rand = random.sample(range(0,V),mask_joint_num)
            mask[i][rand] = 0.0
        mask = mask.reshape(1,T,1,V,1).repeat(N,1,M,1,C)
        return mask

#random joint mask same for frames
class Jointmask2:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
    def __call__(self, data, frame):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        mask_joint_num = int(V*self.mask_ratio)
        rand = random.sample(range(0,V),mask_joint_num)
        mask[:,:frame,rand] = 0.0
        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        return mask


if __name__ =='__main__':
    J = Jointmask(0.5,1,10,1)
    T = TemporalMask(0.5,1,10,1)
    mask = J(torch.rand((1,1,3,10,1)),3)
    print(mask)
    