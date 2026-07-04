from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
from natsort import natsorted
from glob import glob
import pickle

from transformers import AutoProcessor
#正常的catagory list
#CATEGORY_LIST = [ 'airplane', 'ape', 'bat', 'bear', 'bicycle', 'butterfly', 'cabin', 'camel', 'cannon', 'car_(sedan)', 'castle', 'cat', 'chicken', 'church', 'cow', 'crocodilian', 'deer', 'dog', 'dolphin', 'elephant', 'fish', 'frog', 'giraffe', 'hedgehog', 'helicopter', 'horse', 'hotdog', 'jellyfish', 'kangaroo', 'lion', 'lobster', 'motorcycle', 'mouse', 'owl', 'parrot', 'sailboat', 'sea_turtle', 'seal', 'shark', 'sheep', 'skyscraper', 'spider', 'starfish', 'tank', 'teddy_bear', 'tiger', 'zebra']

#sub003的list
CATEGORY_LIST = ['airplane', 'ape', 'bat', 'bear', 'bicycle', 'cabin', 'camel', 'cannon', 'car_(sedan)', 'castle', 'cat', 'chicken', 'church', 'cow', 'crocodilian', 'deer', 'dog', 'dolphin', 'elephant', 'fish', 'frog', 'giraffe', 'hedgehog', 'helicopter', 'horse', 'hotdog', 'jellyfish', 'kangaroo', 'lion', 'lobster', 'motorcycle', 'mouse', 'owl', 'parrot', 'sailboat', 'sea_turtle', 'seal', 'shark', 'sheep', 'skyscraper', 'spider', 'starfish', 'tank', 'teddy_bear', 'tiger', 'zebra','47']

#sub022的list
CATEGORY_LIST = ['airplane', 'ape', 'bat', 'bear', 'bicycle', 'butterfly', 'cabin', 'camel', 'cannon', 'car_(sedan)', 'castle', 'cat', 'chicken', 'church', 'cow', 'crocodilian', 'deer', 'dog', 'dolphin', 'elephant', 'fish', 'frog', 'giraffe', 'hedgehog', 'helicopter', 'horse', 'hotdog', 'jellyfish', 'kangaroo', 'lion', 'lobster', 'motorcycle', 'mouse', 'owl', 'parrot', 'sailboat', 'sea_turtle', 'seal', 'shark', 'sheep', 'skyscraper', 'spider', 'starfish', 'tank', 'teddy_bear', 'tiger', 'zebra', '47']

EMOTION_LIST = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img



#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore


def get_classify_labels(data_path):
    file_name = data_path.split('/')[-1]
    name_split = file_name.split('_')
    emotion = name_split[3]
    category = '_'.join(name_split[4:-2])

    return emotion, category


class eeg_pretrain_dataset(Dataset):
    def __init__(self, path='/home/dell/脑电数据_250529/', roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True):
        super(eeg_pretrain_dataset, self).__init__()
        data = []
        images = []
        self.input_paths = [str(f) for f in sorted(Path(path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]

        assert len(self.input_paths) != 0, 'No data found'
        self.data_len  = 512 #固定时间长度
        self.data_chan = 64 #固定通道数

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        data_path = self.input_paths[index]

        emotion, category = get_classify_labels(data_path)
        print(data_path, emotion, category)
        emotion_idx = EMOTION_LIST.index(emotion)
        category_idx = CATEGORY_LIST.index(category)
        emotion_label = torch.tensor(emotion_idx, dtype=torch.long)
        category_label = torch.tensor(category_idx, dtype=torch.long)

        data = np.load(data_path)



#时间轴处理
        #如果原始数据的时间点数 > 目标时间点数 (512)
        if data.shape[-1] > self.data_len:

            # 进行随机裁剪 (random cropping)
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)
            # 从随机选择的起始点开始，裁剪出目标长度的时间段
            data = data[:, idx: idx+self.data_len]

        # 如果原始数据的时间点数小于或等于目标时间点数    
        else:
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)


        ret = np.zeros((self.data_chan, self.data_len))#初始化一个全零的NumPy 数组，形状为 (128, 512)


#通道数处理
        # 如果目标通道数 (128) >原始数据通道数 (64)
        if (self.data_chan > data.shape[-2]):
            # 进行通道重复填充 
            for i in range((self.data_chan//data.shape[-2])):
                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            
            if self.data_chan % data.shape[-2] != 0:
                ret[ -(self.data_chan%data.shape[-2]):, :] = data[: (self.data_chan%data.shape[-2]), :]

        # 如果目标通道数 (128) < 原始数据通道数
        elif(self.data_chan < data.shape[-2]):
            # 进行随机通道裁剪
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]

        # 如果目标通道数等于原始通道数   
        # print(ret.shape)
        elif(self.data_chan == data.shape[-2]):
            ret = data# 直接使用原始数据

        # ret = ret/10 # 原版 reduce an order
        ret = ret * 1e6 # 修复1 修复数据被缩小的问题（数据集缩小+代码缩小）
        # ret = ret * 1e3 # 理论上应该完美 灵机一动
        # ret = ret * 1e5 #基于代码里原来有缩小10倍，现在才是真正扩大1e6

        # torch.tensor()
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret, 'emotion': emotion_label, 'category': category_label} #,


def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs


def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

EEG_EXTENSIONS = [
    '.mat'
]


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in EEG_EXTENSIONS)


def make_dataset(dir):

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, topdown=False)):#
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

from PIL import Image
import numpy as np
 


class EEGDataset_r(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, image_transform=identity):

        self.imagenet = '/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/train/'
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        # # Compute size
        self.size = 100

    # Get size
    def __len__(self):
        return 100

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = torch.randn(128,512)

        # print(image.shape)
        label = torch.tensor(0).long()
        image = torch.randn(3,512,512)
        image_raw = image

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}


class EEGDataset_s(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, image_transform=identity):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = '/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/train/'
        self.image_transform = image_transform
        self.num_voxels = 440
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()

        eeg = eeg[20:460,:]

        # Get label
        image_name = self.images[self.data[i]["image"]]
        # image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        return image_name



class EEGDataset(Dataset):

    def __init__(self, eeg_signals_path, image_transform=identity, subject = 0):#加载 ID 为 4 的受试者数据
        
        # 使用 PyTorch 的 torch.load() 函数来加载指定路径 (eeg_signals_path) 的 .pth 文件
        loaded = torch.load(eeg_signals_path)

        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']        

        self.labels = loaded["labels"]
        self.images_list = loaded["images"]#草图和图片共用的
        self.subjects = loaded["subject"]
        self.emotions_list = loaded["emotion"] 
        self.category_list = loaded["category"]


        #修改self.imagenet变量为图片数据集根目录
        self.imagenet = '/home/dell/dataset_for_sketchEEG/emotion_image/'
        #草图根目录
        self.sketch_root = '/home/dell/dataset_for_sketchEEG/emotion_sketch/'

        self.image_transform = image_transform
        self.num_voxels = 440  
        self.data_len = 512  #模型预期输入的EEG信号的时间点长度是 512
        # Compute size
        self.size = len(self.data)

        self.processor = AutoProcessor.from_pretrained("/home/dell/dreamdiffusion_project/DreamDiffusion/pretrains/clip-vit-large-patch14")
    
    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        # print(self.data[i])

        #修改2：将 NumPy 数组转换为 PyTorch 张量，再对其float() 和 t() 
        #eeg = self.data[i]["eeg"].float().t()
        eeg = torch.from_numpy(self.data[i]["eeg"]).float().t()

        eeg = eeg[20:460,:]
        ##### 2023 2 13 add preprocess and transpose
        eeg = np.array(eeg.transpose(0,1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        ##### 2023 2 13 add preprocess
        label = torch.tensor(self.data[i]["label"]).long()

        # Get label
        label_idx = self.data[i]["label"]

        emotion_idx = self.data[i]["emotion"]
        emotion_folder = self.emotions_list[emotion_idx]

        category_idx = self.data[i]["category"]
        image_category_name = self.category_list[category_idx]

        image_idx = self.data[i]["image"]
        image_id_string = self.images_list[image_idx]

        #构建文件名（图片/草图共用，不含扩展名）
        base_filename = f"{image_category_name}_{image_id_string}"

        #构建图片路径
        image_filename = f"{base_filename}.jpg"
        image_path = os.path.join(self.imagenet, emotion_folder, image_filename)
        print(f"图片路径: {image_path}")

        #构建草图路径
        sketch_filename = f"{base_filename}.png"
        sketch_path = os.path.join(self.sketch_root, emotion_folder, sketch_filename)
        print(f"草图路径: {sketch_path}")


        image_raw = Image.open(image_path).convert('RGB') 
        image = np.array(image_raw) / 255.0
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

        #reset 成512*512

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw,
                'sketch_path': sketch_path,'emotion_idx': emotion_idx, 'category_idx': category_idx}


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)

        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        #修改1 'eeg' 数据第二个维度450-600的限制
        #self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        self.split_idx = [i for i in self.split_idx if i < len(self.dataset.data)]

        # Compute size
        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

# Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(
            eeg_signals_path='/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_sketches_alpha_raw_sub033.pth', 
            splits_path = '/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/train_val_test_sibling_group4.pth',
            # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth',

            image_transform=identity, subject = 0):
    # if subject == 0:
        # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth'
    
    #检查传入的image_transform是否是一个列表
    if isinstance(image_transform, list):
        dataset_train = EEGDataset(eeg_signals_path, image_transform[0], subject )
        dataset_test = EEGDataset(eeg_signals_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset(eeg_signals_path, image_transform, subject)
        dataset_test = EEGDataset(eeg_signals_path, image_transform, subject)
    print('dataset_train len', len(dataset_train))
    split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train', subject= subject)
    split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test', subject = subject)
    return (split_train, split_test)



def create_EEG_dataset_r(eeg_signals_path='../dreamdiffusion/datasets/eeg_5_95_std.pth', 
            # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_single.pth',
            splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth',
            image_transform=identity):
    if isinstance(image_transform, list):
        dataset_train = EEGDataset_r(eeg_signals_path, image_transform[0])
        dataset_test = EEGDataset_r(eeg_signals_path, image_transform[1])
    else:
        dataset_train = EEGDataset_r(eeg_signals_path, image_transform)
        dataset_test = EEGDataset_r(eeg_signals_path, image_transform)
    # split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train')
    # split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test')
    return (dataset_train,dataset_test)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img
def normalize2(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img
def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')
if __name__ == '__main__':
    import scipy.io as scio
    import copy
    import shutil


if __name__ == '__main__':
    dataset_pretrain = eeg_pretrain_dataset(path='/home/dell/dreamdiffusion_project/DreamDiffusion/脑电数据_250529')
    input_paths = dataset_pretrain.input_paths
    for input_path in input_paths:
        print(input_path, get_classify_labels(input_path))
    # print(input_paths)