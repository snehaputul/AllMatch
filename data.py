import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import copy
import random
import torch
from skimage import filters


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    if partition == 'validate':
        partition = 'train'
    if partition == 'validate_train':
        partition = 'train'
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random() * max_dropout_ratio  
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


def random_scale(pointcloud, scale_low=0.8, scale_high=1.25):
    N, C = pointcloud.shape
    scale = np.random.uniform(scale_low, scale_high)
    pointcloud = pointcloud * scale
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points=1024, partition='train', data_split='labeled', perceptange=10,
                 balanced_split=True, args=None):
        data, label = load_data(partition)
        # count instances of each classes in the label
        unique, counts = np.unique(label, return_counts=True)
        print(unique, counts)

        self.num_points = num_points
        self.partition = partition
        if args is None:
            raise ValueError('args is required')
        self.args = args

        if not balanced_split:
            if self.partition == 'train':
                labeled_sample_num = int(len(label) * perceptange / 100.0)
                unlabeled_sample_num = len(label) - labeled_sample_num

                if data_split == 'labeled':
                    self.data, self.label = data[unlabeled_sample_num:, :, :], label[unlabeled_sample_num:]
                else:
                    self.data, self.label = data[:unlabeled_sample_num, :, :], label[:unlabeled_sample_num]

            elif self.partition == 'validate':
                labeled_sample_num = int(len(label) * perceptange / 100.0)
                unlabeled_sample_num = len(label) - labeled_sample_num
                self.data, self.label = data[:unlabeled_sample_num, :, :], label[:unlabeled_sample_num]
            else:
                self.data, self.label = data, label
        else:
            labeled_sample_num = int(len(label) * perceptange / 100.0)
            samples_per_class = int(np.ceil(labeled_sample_num / unique.shape[0]))
            try:
                select_indices = []
                for i in range(unique.shape[0]):
                    select_indices.extend(np.random.choice(np.where(label == unique[i])[0], samples_per_class, replace=False))
            except:
                # select randomly
                select_indices = np.random.choice(np.arange(label.shape[0]), labeled_sample_num, replace=False)
            # all sampeles except the ones in labelled set
            unlabelled_sample_indices = np.setdiff1d(np.arange(label.shape[0]), select_indices)

            if self.partition == 'train':
                if data_split == 'labeled':
                    self.data, self.label = data[select_indices, :, :], label[select_indices]
                else:
                    self.data, self.label = data[unlabelled_sample_indices, :, :], label[unlabelled_sample_indices]
            else:
                self.data, self.label = data, label

        self.easy_mask = torch.zeros(len(self.data))
        self.history_loss = torch.zeros(len(self.data))

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud)  # open for dgcnn not for our idea  for all
            pt = copy.deepcopy(pointcloud)
            non_aug_pt = pt

            pointcloud = translate_pointcloud(pointcloud)
            if self.easy_mask[item] == False:
                aug_num = np.random.randint(2, high=5)
                aug_list = random.sample(range(4), aug_num)
                pointcloud_strongaug = pt
                if 0 in aug_list:
                    pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                if 1 in aug_list:
                    pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                if 2 in aug_list:
                    pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                if 3 in aug_list:
                    pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 4:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 5:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)

            elif self.easy_mask[item] == True and self.args.aug_strength == 6:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 7:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 8:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)

            np.random.shuffle(non_aug_pt)
            np.random.shuffle(pointcloud)
            np.random.shuffle(pointcloud_strongaug)

            return non_aug_pt, pointcloud, pointcloud_strongaug, label, item, self.easy_mask[item]
        else:
            return pointcloud, label, item, self.easy_mask[item]

    def __len__(self):
        return self.data.shape[0]

    def update_loss(self, idx, iter_loss):
        self.history_loss[idx] = 0.001 * self.history_loss[idx] + 0.999 * iter_loss
        if np.nan in self.history_loss:
            print('error')

    def update_mask(self):
        np_loss = self.history_loss.data.numpy()
        np_loss = np.sort(np_loss)

        g1 = g2 = filters.threshold_otsu(np_loss)

        self.easy_mask[self.history_loss < g1] = True
        self.easy_mask[self.history_loss > g2] = False


def load_ScanObjectNN(partition):
    BASE_DIR = 'data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='train', data_split='labeled', perceptange=10, args=None):
        super().__init__()
        data, label = load_ScanObjectNN(partition)
        if args is None:
            raise ValueError("args is required")
        label = label.reshape(-1, 1)
        self.num_points = num_points
        self.partition = partition
        unique, counts = np.unique(label, return_counts=True)

        labeled_sample_num = int(len(label) * perceptange / 100.0)
        samples_per_class = int(np.ceil(labeled_sample_num / unique.shape[0]))
        try:
            select_indices = []
            for i in range(unique.shape[0]):
                select_indices.extend(np.random.choice(np.where(label == unique[i])[0], samples_per_class, replace=False))
        except:
            # select randomly
            select_indices = np.random.choice(np.arange(label.shape[0]), labeled_sample_num, replace=False)
        # all sampeles except the ones in labelled set
        unlabelled_sample_indices = np.setdiff1d(np.arange(label.shape[0]), select_indices)

        if self.partition == 'train':
            if data_split == 'labeled':
                self.data, self.label = data[select_indices, :, :], label[select_indices]
            else:
                self.data, self.label = data[unlabelled_sample_indices, :, :], label[unlabelled_sample_indices]
        else:
            self.data, self.label = data, label

        print("ScanObject NN data size: ", self.data.shape)
        unique, counts = np.unique(self.label, return_counts=True)
        print(unique, counts)

        self.easy_mask = torch.zeros(len(self.data))
        self.history_loss = torch.zeros(len(self.data))

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud)  
            pt = copy.deepcopy(pointcloud)
            non_aug_pt = pt

            pointcloud = translate_pointcloud(pointcloud)
            if not self.easy_mask[item]:
                aug_num = np.random.randint(2, high=5)
                aug_list = random.sample(range(4), aug_num)
                pointcloud_strongaug = pt
                if 0 in aug_list:
                    pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                if 1 in aug_list:
                    pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                if 2 in aug_list:
                    pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                if 3 in aug_list:
                    pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 4:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 5:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)

            elif self.easy_mask[item] == True and self.args.aug_strength == 6:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 7:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)

            elif self.easy_mask[item] == True and self.args.aug_strength == 8:
                pointcloud_strongaug = pt
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)

            np.random.shuffle(non_aug_pt)
            np.random.shuffle(pointcloud)
            np.random.shuffle(pointcloud_strongaug)

            return non_aug_pt, pointcloud, pointcloud_strongaug, label, item
        else:
            return pointcloud, label, item

    def __len__(self):
        return self.data.shape[0]

    def update_loss(self, idx, iter_loss):
        self.history_loss[idx] = 0.001 * self.history_loss[idx] + 0.999 * iter_loss
        if np.nan in self.history_loss:
            print('error!')

    def update_mask(self):
        np_loss = self.history_loss.data.numpy()
        np_loss = np.sort(np_loss)

        g1 = g2 = filters.threshold_otsu(np_loss)

        self.easy_mask[self.history_loss < g1] = True
        self.easy_mask[self.history_loss > g2] = False

