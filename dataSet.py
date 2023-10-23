import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_utils.init_data import check_data
from transforms import GetROI, MyPad
from torchvision.transforms.functional import resize


class YanMianDataset(Dataset):
    def __init__(self, root: str, transforms=None, data_type: str = 'train', ki=-1, k=5, json_path=None,
                 mask_path=None, txt_path=None):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        self.root = os.path.join(root, 'datas')
        self.transforms = transforms
        self.json_list = []
        self.data_type = data_type
        self.run_env = '/' if '/data/lk' in os.getcwd() else '\\'

        # read txt file and save all json file list (train/val/test)
        if json_path is None:
            json_path = os.path.join(self.root, 'jsons')
        if txt_path is None:
            txt_path = os.path.join(self.root.replace('datas', 'data_utils'), data_type + '.txt')
        if mask_path is not None:
            self.mask_path = mask_path
        else:
            self.mask_path = None
        assert os.path.exists(txt_path), 'not found {} file'.format(data_type + '.txt')

        with open(txt_path) as read:
            txt_path = [line.strip() for line in read.readlines() if len(line.strip()) > 0]
        if ki != -1:
            # 使用k折交叉验证
            assert data_type in ['train', 'val'], 'test can not use cross validation'
            random.seed(1)
            random.shuffle(txt_path)
            length = len(txt_path) // k
            if data_type == 'val':
                txt_path_ = txt_path[length * ki:length * (ki + 1)]
            else:
                txt_path_ = txt_path[:length * ki] + txt_path[length * (ki + 1):]
            self.json_list = [os.path.join(json_path, i) for i in txt_path_]
        else:
            self.json_list = [os.path.join(json_path, i) for i in txt_path]

        # check file
        assert len(self.json_list) > 0, 'in "{}" file does not find any information'.format(data_type + '.txt')
        for json_dir in self.json_list:
            assert os.path.exists(json_dir), 'not found "{}" file'.format(json_dir)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        img_root = os.path.join(self.root, 'images')
        if self.mask_path is None:
            self.mask_path = os.path.join(self.root, 'masks')

        # load json data
        json_dir = self.json_list[index]
        json_str = open(json_dir, 'r', encoding='utf-8')
        json_data = json.load(json_str)
        json_str.close()

        # get image
        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join(img_root, img_name)
        origin_image = Image.open(img_path)
        # 转换为灰度图，再变为三通道
        img = origin_image.convert('L')
        img = img.convert('RGB')

        # get landmark data
        landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
        landmark = {i['Label']: np.array(i['Position'])[:2] for i in landmark}
        target = {'landmark': landmark, 'data_type': self.data_type, 'img_name': img_name}

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


if __name__ == '__main__':
    # from transforms import RightCrop
    d = os.getcwd()
    import transforms as T
    mean = (0.12888692, 0.12888692, 0.12888692)
    std = (0.16037938, 0.16037938, 0.16037938)
    trans = T.Compose([
        T.RandomResize(int(0.8 * 256), 256, resize_ratio=1, shrink_ratio=1),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        # T.RandomRotation(10, rotate_ratio=0.7, expand_ratio=0.7),
        T.GenerateHeatmap(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.MyPad(256)
    ])
    mydata = YanMianDataset(d, data_type='test', transforms=trans)
    # a,b = mydata[0]
    # c =1
    for i in range(len(mydata)):
        a,b = mydata[i]
        print(i)


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
