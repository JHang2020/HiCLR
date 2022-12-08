import pickle
import numpy as np
from tqdm import tqdm

# Linear
print('-' * 20 + 'Linear Eval' + '-' * 20)

joint_path = '/mnt/netdisk/zhangjh/Code/AimCLR_loss/work_dir/PKU2/xsub/joint_MutalDDM_with4parallel_ablmask_1500/linear_tmp/'
bone_path = '/mnt/netdisk/zhangjh/Code/AimCLR_loss/work_dir/PKU2/xsub/bone_MutalDDM_with4parallel_ablmask_1500/linear_tmp/'
motion_path = '/mnt/netdisk/zhangjh/Code/AimCLR_loss/work_dir/PKU2/xsub/motion_MutalDDM_with4parallel_ablmask_1500/linear_tmp/'
 
 
label = open('/mnt/netdisk/linlilang/PKUMMD2/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(joint_path + 'test_result.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(bone_path + 'test_result.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(motion_path + 'test_result.pkl', 'rb')
r3 = list(pickle.load(r3).items())

alpha1 = [0.5,0.5,0.5]# average fusion
alpha2 = [0.6, 0.6, 0.4] # aimclr weighted fusion
if 1:
    best = 0.0
    best_weight = []
    for alpha in [alpha1,alpha2]:
        right_num = total_num = right_num_5 = 0
        for i in tqdm(range(len(label[0]))):
            _, l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        if acc> best:
            best = acc
            best_weight = alpha
        
        print(alpha, 'top1: ', acc)

    print(best_weight, best)