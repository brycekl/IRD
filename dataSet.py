import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from transforms import GetROI, MyPad
from torchvision.transforms.functional import resize

posture_label = {'PP': 9, 'SP': 10, 'ST': 11}
position_label = {'B3': 12, 'U3': 13, 'U5': 14, 'UE': 15}
landmark_label = {'left': 6, 'right': 5}   # x轴上6<5
seg_label = {'left': 8, 'right': 7}


class IRDDataset(Dataset):
    def __init__(self, data_type: str = 'train', position_type='4-all', task='landmark', transforms=None, ki=-1, k=5):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        self.data_root = '../datas/IRD/COCO_style'
        self.transforms = transforms
        self.data_type = data_type
        self.position_type = position_type
        self.task = task
        self.run_env = '/' if '/data/lk' in os.getcwd() else '\\'
        self.wrong = {'landmark': []}
        self.legal_wrong = {'landmark': []}
        with open('./data_utils/landmark_anno_error', 'r') as reader:
            for line in reader.readlines():
                self.legal_wrong['landmark'].append(line.strip())

        # load position type and data type data from json file
        with open('data_utils/data.json', 'r') as reader:
            self.data_list = json.load(reader)[position_type]
            self.train_info = self.data_list['train_info']
        if ki != -1:
            # 使用k折交叉验证
            assert data_type in ['train', 'val'], 'test can not use cross validation'
            all_json = self.data_list['train'] + self.data_list['val']
            random.seed(1)
            random.shuffle(all_json)
            length = len(all_json) // k
            if data_type == 'val':
                self.data_list = all_json[length * ki:length * (ki + 1)]
            else:
                self.data_list = all_json[:length * ki] + all_json[length * (ki + 1):]
        else:
            self.data_list = self.data_list[data_type]

        # check file
        assert len(self.data_list) > 0, 'in "{}" file does not find data.'.format(position_type + '_' + data_type)
        for data_name in self.data_list:
            assert os.path.exists(os.path.join(self.data_root, 'jsons', data_name + '.json')), f'not found {data_name} file'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        base_name = self.data_list[index]
        img, landmark, poly_mask = get_name_data(self.data_root, base_name)

        target = {'landmark': landmark, 'poly_mask': poly_mask, 'data_type': self.data_type, 'img_name': base_name,
                  'origin_landmark': landmark, 'transforms': [], 'h_flip': False}

        check_data(target, base_name, self.wrong, self.legal_wrong)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):  # 如何取样本，实现自定义的batch输出
        images, targets = list(zip(*batch))  # batch里每个元素表示一张图片和一个gt
        batched_imgs = cat_list(images, fill_value=0)  # 统一batch的图片大小
        mask = [i['mask'] for i in targets]
        batched_targets = {'mask': cat_list(mask, fill_value=255)}
        for item in targets[0]:
            if item not in ['mask']:
                batched_targets[item] = [i[item] for i in targets]
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))  # 获取每个维度的最大值
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)  # batch, c, h, w
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def check_data(target, name, wrong, legal_wrong):
    if 'landmark' in target:
        landmark = target['landmark']
        if landmark[5][0] > landmark[6][0] and name not in legal_wrong['landmark']:
            wrong['landmark'].append(name)


def get_name_data(data_root, name):
    # load json data
    with open(os.path.join(data_root, 'jsons', name + '.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # get image
    img_path = os.path.join(data_root, 'images', name + '.png')
    img = Image.open(img_path).convert('RGB')

    # get landmark data
    landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    landmark = {i['Label']: np.array(i['Position'])[:2] for i in landmark}

    # get mask
    mask_path = os.path.join(data_root, 'masks', name + '_255.png')
    poly_mask = Image.open(mask_path)

    return img, landmark, poly_mask


if __name__ == '__main__':
    # from transforms import RightCrop
    d = os.getcwd()
    import transforms as T
    mean = (0.12888692, 0.12888692, 0.12888692)
    std = (0.16037938, 0.16037938, 0.16037938)
    base_size = 256
    trans = T.Compose([
        # T.RandomResize(int(0.8 * base_size), base_size),
        # T.RandomResize(int(base_size*0.8), base_size, resize_ratio=1, shrink_ratio=1),
        T.RandomResize(base_size, base_size, resize_ratio=1, shrink_ratio=0),
        # T.Resize([base_size]),
        T.RandomHorizontalFlip(1),
        T.RandomVerticalFlip(1),
        T.GenerateMask(task='all'),
        # T.ToTensor(),
        # T.Normalize(mean=mean, std=std),
        T.MyPad([base_size])
    ])
    mydata = IRDDataset(data_type='train', transforms=trans)
    # a,b = mydata[0]
    # c =1
    for i in range(len(mydata)):
        img, target = mydata[i]
        print(i, target['img_name'])
    s = 1
