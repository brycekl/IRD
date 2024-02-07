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


class IRDDataset(Dataset):
    def __init__(self, data_type: str = 'train', position_type='4-all', transforms=None, ki=-1, k=5):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        self.data_root = '../datas/IRD/COCO_style'
        self.transforms = transforms
        self.data_type = data_type
        self.position_type = position_type
        self.run_env = '/' if '/data/lk' in os.getcwd() else '\\'

        # load position type and data type data from json file
        with open('data_utils/data.json', 'r') as reader:
            self.json_list = json.load(reader)[position_type]
            self.train_info = self.json_list['train_info']
        if ki != -1:
            # 使用k折交叉验证
            assert data_type in ['train', 'val'], 'test can not use cross validation'
            all_json = self.json_list['train'] + self.json_list['val']
            random.seed(1)
            random.shuffle(all_json)
            length = len(all_json) // k
            if data_type == 'val':
                self.json_list = all_json[length * ki:length * (ki + 1)]
            else:
                self.json_list = all_json[:length * ki] + all_json[length * (ki + 1):]
        else:
            self.json_list = self.json_list[data_type]
        self.json_list = [os.path.join(self.data_root, 'jsons', i) for i in self.json_list]

        # check file
        assert len(self.json_list) > 0, 'in "{}" file does not find data.'.format(position_type + '_' + data_type)
        for json_dir in self.json_list:
            assert os.path.exists(json_dir), 'not found "{}" file'.format(json_dir)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        # load json data
        json_dir = self.json_list[index]
        with open(json_dir, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # get image
        img_name = os.path.basename(json_dir).split('.json')[0] + '.jpg'
        img_path = os.path.join(self.data_root, 'images', img_name)
        img = Image.open(img_path).convert('RGB')

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
