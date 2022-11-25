import random
from matplotlib import use
import numpy as np
import pickle, torch
from . import tools


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._aug(data_numpy)
        data3 = self._aug(data_numpy)
        data4 = self._strong_aug(data_numpy)
        data5 = self._strong_aug(data_numpy)
        #data5 = data4
        return [data1, data2, data3, data4, data5], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy

class Feeder_triple_abl(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._normal_aug(self._basic_aug(data_numpy))
        data2 = self._normal_aug(self._basic_aug(data_numpy))
        data3 = self._normal_aug(self._basic_aug(data_numpy))
        data4 = self._normal_aug(self._basic_aug(data_numpy))
        data5 = self._normal_aug(self._basic_aug(data_numpy))
        #data5 = data4
        return [data1, data2, data3, data4, data5], label

    def _basic_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    # you can choose different combinations
    def _normal_aug(self, data_numpy):
        #if self.temperal_padding_ratio > 0:
        #    data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        #if self.shear_amplitude > 0:
        #    data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy


class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy

class Feeder_semieval(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mode='finetune',
                 data_view='joint',
                 label_percent=0.1,
                 norm=False,
                 use_aug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        if 'NTU' in self.data_path or 'ntu' in self.data_path:
            self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.label_percent = label_percent
        self.aug = use_aug
        self.norm = norm 
        self.mode = mode
        self.data_view = data_view
        self.load_data(mmap)
        if norm:
            self.get_mean_map()
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)#shape C,1,V,1
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))#shape C 1 V 1 
        self.std_map[self.std_map<1e-10] = 1e-10
        print('end...')
    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        
        if self.data_view=='motion':
            motion1 = np.zeros_like(self.data)
            motion1[:, :, :-1, :, :] = self.data[:, :, 1:, :, :] - self.data[:, :, :-1, :, :]
            self.data = motion1
            print('Motion view is used...')
        
        if self.data_view=='bone':
            bone1 = np.zeros_like(self.data)
            for v1, v2 in self.Bone:
                bone1[:, :, :, v1 - 1, :] = self.data[:, :, :, v1 - 1, :] - self.data[:, :, :, v2 - 1, :]
            self.data = bone1
            print('Bone view is used...')
        
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        n = len(self.label)
        self.ori_n = n
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return self.ori_n
    def _aug(self, data_numpy):
        self.temperal_padding_ratio = 6
        self.shear_amplitude = 0.5
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    def __getitem__(self, index):
        # get data
        index = index%len(self.label)
        data_numpy = np.array(self.data[index])
        if self.mode == 'finetune':
            label = self.label[index]
        
        # processing
        if self.norm:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.aug:
            data_numpy = self._aug(data_numpy)
        
        #print(data_numpy[:,0,0,0])
        if self.mode == 'finetune':
            return data_numpy, label
        else:
            return data_numpy, torch.ones((self.T,))
    def top_k(self, score, top_k):
        rank = score.argsort()#将x中的元素从小到大排列，提取其对应的index
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]#是01数组，1表示truth_label
                                                                             #在前top_k大的预测得分中
        return sum(hit_top_k) * 1.0 / len(hit_top_k)#返回accurracy

class Feeder_finetune(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mode='finetune',
                 data_view='joint',
                 use_aug=False,
                 norm=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        if 'NTU' in self.data_path or 'ntu' in self.data_path:
            self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.norm = norm 
        self.mode = mode
        self.aug=use_aug
        self.data_view = data_view
        self.load_data(mmap)
        if norm:
            self.get_mean_map()
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)#shape C,1,V,1
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))#shape C 1 V 1 
        self.std_map[self.std_map<1e-10] = 1e-10
        print('end...')
    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        
        if self.data_view=='motion':
            motion1 = np.zeros_like(self.data)
            motion1[:, :, :-1, :, :] = self.data[:, :, 1:, :, :] - self.data[:, :, :-1, :, :]
            self.data = motion1
            print('Motion view is used...')
        
        if self.data_view=='bone':
            bone1 = np.zeros_like(self.data)
            for v1, v2 in self.Bone:
                bone1[:, :, :, v1 - 1, :] = self.data[:, :, :, v1 - 1, :] - self.data[:, :, :, v2 - 1, :]
            self.data = bone1
            print('Bone view is used...')
        
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)
    def _aug(self, data_numpy):
        self.temperal_padding_ratio = 6
        self.shear_amplitude = 0.5
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        if self.mode == 'finetune':
            label = self.label[index]
        
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.norm:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.aug:
            data_numpy = self._aug(data_numpy)
        #print(data_numpy[:,0,0,0])
        if self.mode == 'finetune':
            return data_numpy, label
        else:
            return data_numpy, torch.ones((self.T,))
    def top_k(self, score, top_k):
        rank = score.argsort()#将x中的元素从小到大排列，提取其对应的index
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]#是01数组，1表示truth_label
                                                                             #在前top_k大的预测得分中
        return sum(hit_top_k) * 1.0 / len(hit_top_k)#返回accurracy
