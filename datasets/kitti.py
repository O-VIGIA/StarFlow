# from ast import main
# from mimetypes import init
import sys, os
import os.path as osp
# from unicodedata import name
import numpy as np
import glob
import torch.utils.data as data

from transforms import transforms

__all__ = ['KITTI','Kitti_Occlusion', 'LidarKITTI', 'SFKITTI']


class KITTI(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True):
        self.root = osp.join(data_root, 'kitti_processed')
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]

        return pc1, pc2

class Kitti_Occlusion(data.Dataset):
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True,npoints=8192):
        self.npoints = npoints
        
        
        self.root = osp.join(data_root, 'kitti_rm_ground')

        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground
        
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data["pos1"][:, (1, 2, 0)]
                pos2 = data["pos2"][:, (1, 2, 0)]
                flow = data["gt"][:, (1, 2, 0)]
                
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        loc1 = pos1[:,2] < 35
        pos1 = pos1[loc1]
        
        flow = flow[loc1]
        
        loc2 = pos2[:,2] < 35
        pos2 = pos2[loc2]
        
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

        pos1_ = np.copy(pos1)[sample_idx1, :]
        pos2_ = np.copy(pos2)[sample_idx2, :]
        flow_ = np.copy(flow)[sample_idx1, :]


        color1 = np.zeros([self.npoints, 3])
        color2 = np.zeros([self.npoints, 3])
        
        mask = np.ones([self.npoints])

        return pos1_, pos2_, color1, color2, flow_, mask

    def __len__(self):
        return len(self.datapath)

class LidarKITTI(data.Dataset):
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True,
                 npoints=8192):
        
        self.npoints = npoints
        
        self.root = osp.join(data_root, 'lidar_kitti')

        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground
        
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pc1']
                pos2 = data['pc2']
                if 'flow' in data:
                    flow = data['flow']
                else:
                    print(fn + "is no flow gt !")
                    flow = np.zeros_like(pos1)
                # pos1 = data["pos1"][:, (1, 2, 0)]
                # pos2 = data["pos2"][:, (1, 2, 0)]
                # flow = data["gt"][:, (1, 2, 0)]
                
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        # loc1 = pos1[:,2] < 35
        # pos1 = pos1[loc1]
        
        # flow = flow[loc1]
        
        # loc2 = pos2[:,2] < 35
        # pos2 = pos2[loc2]
        
        # --------------cut ground---------------------#
        is_not_ground_s = (pos1[:, 1] > -1.4)
        is_not_ground_t = (pos2[:, 1] > -1.4)

        pos1 = pos1[is_not_ground_s,:]
        flow = flow[is_not_ground_s,:]
        pos2 = pos2[is_not_ground_t,:]
        
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

        
        # if n1 > self.num_points:
        #     sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
        # else:
        #     sample_idx1 = np.random.choice(n1, n1, replace=False)

        # if n2 > self.num_points:
        #     sample_idx2 = np.random.choice(n2, self.num_points, replace=False)
        # else:
        #     sample_idx2 = np.random.choice(n2, n2, replace=False)
        
        pos1_ = np.copy(pos1)[sample_idx1, :]
        pos2_ = np.copy(pos2)[sample_idx2, :]
        flow_ = np.copy(flow)[sample_idx1, :]


        # color1 = np.zeros([self.npoints, 3])
        # color2 = np.zeros([self.npoints, 3])
        
        norm1 = pos1_
        norm2 = pos2_
        
        mask = np.ones([self.npoints])

        return pos1_, pos2_, norm1, norm2, flow_, mask

    def __len__(self):
        return len(self.datapath)


class SFKITTI(data.Dataset):
    def __init__(self, train, transform, num_points, data_root, full=True):
        self.root = osp.join(data_root, "sf-kitti")
        print(self.root)
        self.train = train
        self.transform = transform
        self.num_points = num_points

        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, "TRAIN*.npz"))
        else:
            self.datapath = glob.glob(os.path.join(self.root, "TEST*.npz"))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [
            d for d in self.datapath
        ]
        ######

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):

        if index in self.cache:
            pos1, pos2, color1, color2, flow, fg_mask = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, "rb") as fp:
                data = np.load(fp)
                pos1 = data["pc1"]
                pos2 = data["pc2"]
                color1 = pos1
                color2 = pos2
                flow = data["gt"]
                fg_mask = data["fg_index"]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, fg_mask)

         # --------------cut ground ---------------------#
        # is_not_ground_s = (pos1[:, 1] > -1.4)
        # is_not_ground_t = (pos2[:, 1] > -1.4)

        # pos1 = pos1[is_not_ground_s,:]
        # flow = flow[is_not_ground_s,:]
        # pos2 = pos2[is_not_ground_t,:]
        
        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.num_points, replace=False)

            pos1_ = np.copy(pos1[sample_idx1, :])
            pos2_ = np.copy(pos2[sample_idx2, :])
            color1_ = np.copy(color1[sample_idx1, :])
            color2_ = np.copy(color2[sample_idx2, :])
            flow_ = np.copy(flow[sample_idx1, :])
            fg_mask_ = np.copy(fg_mask[sample_idx1])
        else:
            pos1_ = np.copy(pos1[: self.num_points, :])
            pos2_ = np.copy(pos2[: self.num_points, :])
            color1_ = np.copy(color1[: self.num_points, :])
            color2_ = np.copy(color2[: self.num_points, :])
            flow_ = np.copy(flow[: self.num_points, :])
            fg_mask_ = np.copy(fg_mask[: self.num_points])

        return pos1_, pos2_, color1_, color2_, flow_, fg_mask_
