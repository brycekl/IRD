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
    def __init__(self, data_type: str = 'train', position_type='4-all', transforms=None, ki=-1, k=5):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        self.data_root = '../datas/IRD/COCO_style'
        self.transforms = transforms
        self.data_type = data_type
        self.position_type = position_type
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
        # load json data
        base_name = self.data_list[index]
        with open(os.path.join(self.data_root, 'jsons', base_name + '.json'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # get image
        img_path = os.path.join(self.data_root, 'images', base_name + '.jpg')
        img = Image.open(img_path).convert('RGB')

        # get landmark data
        landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
        landmark = {i['Label']: np.array(i['Position'])[:2] for i in landmark}

        # get mask
        mask_path = os.path.join(self.data_root, 'masks', base_name + '_255.jpg')
        mask = Image.open(mask_path)

        target = {'landmark': landmark, 'mask': mask, 'data_type': self.data_type, 'img_name': base_name}

        check_data(target, base_name, self.wrong, self.legal_wrong)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):  # 如何取样本，实现自定义的batch输出
        images, targets = list(zip(*batch))  # batch里每个元素表示一张图片和一个gt
        batched_imgs = cat_list(images, fill_value=0)  # 统一batch的图片大小
        mask = [i['mask'] for i in targets]
        batched_targets = {'landmark': [i['landmark'] for i in targets]}
        batched_targets['img_name'] = [i['img_name'] for i in targets]
        batched_targets['mask'] = cat_list(mask, fill_value=255)
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


if __name__ == '__main__':
    # from transforms import RightCrop
    d = os.getcwd()
    import transforms as T
    mean = (0.12888692, 0.12888692, 0.12888692)
    std = (0.16037938, 0.16037938, 0.16037938)
    base_size = 256
    trans = T.Compose([
        # T.RandomResize(int(0.8 * base_size), base_size),
        T.Resize([base_size]),
        T.RandomHorizontalFlip(1),
        T.RandomVerticalFlip(1),
        T.GenerateHeatmap(),
        # T.ToTensor(),
        # T.Normalize(mean=mean, std=std),
        T.MyPad([base_size])
    ])
    mydata = IRDDataset(data_type='train', transforms=trans)
    # a,b = mydata[0]
    # c =1
    for i in range(len(mydata)):
        img, target = mydata[i]
        print(i)
    s = 1

# train data 1542
# val data 330
# test data 330

# data 1
# 试标 22张不能用
# 1 curve: 5, 5 landmark: 3, 上颌骨（下颌骨）未被标注（无label）:7, 存在曲线未被标注（无label）:7
# data 2
# IMG_20201021_2_55_jpg_Label.json 只标了一条线，且一条线只有一个点
# 0135877_Mengyan_Tang_20200917091142414_jpg_Label.json  未标注曲线
# 1 curve: 3, 5 landmark:6, 上颌骨（下颌骨）未被标注（无label）:17, 存在曲线未被标注（无label）:1
# data 3
# 1 curve: 1, 5 landmark: 5, 上颌骨（下颌骨）未被标注（无label）:2, 存在曲线未被标注（无label）:0
# data 4
# 1 curve: 1, 5 landmark: 5, 上颌骨（下颌骨）未被标注（无label）:9, 存在曲线未被标注（无label）:0
# data 5
# 1 curve: 5, 5 landmark:0, 上颌骨（下颌骨）未被标注（无label）:14, 存在曲线未被标注（无label）:0
# data 6
# 1 curve: 2, 5 landmark:0, 上颌骨（下颌骨）未被标注（无label）:12, 存在曲线未被标注（无label）:0
# 0117667_Yinying_Chen_20210526131902731_jpg_Label
