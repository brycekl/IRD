import math
import random

import cv2
import numpy as np
import torch
from typing import Tuple
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy import ndimage


class SegmentationPresetTrain:
    def __init__(self, input_size, task='landmark', var=40,  max_value=8, stretch=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        trans = []
        trans.extend([
            # RandomResize(min_size, max_size, resize_ratio=1, shrink_ratio=1),
            AffineTransform(rotation=(-15, 15), input_size=input_size, resize_low_high=[0.8, 1], stretch=stretch),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            # RandomRotation(10, rotate_ratio=0.7, expand_ratio=0.7),
            GenerateMask(task=task, var=var, max_value=max_value),
            ToTensor(),
            Normalize(mean=mean, std=std),
            MyPad(input_size)
        ])

        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, input_size, task='landmark', var=40,  max_value=8, stretch=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Compose([
            # RandomResize(base_size, base_size, resize_ratio=1, shrink_ratio=0),
            # Resize([base_size]),
            AffineTransform(input_size=input_size, stretch=stretch),
            GenerateMask(task=task, var=var, max_value=max_value),
            ToTensor(),
            Normalize(mean=mean, std=std),
            MyPad(input_size)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, input_size=(256, 256), task='landmark', var=40, max_value=8, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), stretch=True):
    if train:
        return SegmentationPresetTrain(input_size, task, var, max_value, stretch=stretch, mean=mean, std=std)
    else:
        return SegmentationPresetEval(input_size, task, var, max_value, stretch=stretch, mean=mean, std=std)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None, resize_ratio=1, shrink_ratio=1):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
        self.resize_ratio = resize_ratio  # 进行resize的概率，为1即一定进行resize
        self.shrink_ratio = shrink_ratio

    def __call__(self, image, target):
        if np.random.random() > self.resize_ratio:
            return image, target
        size = random.randint(self.min_size, self.max_size) \
            if np.random.random() < self.shrink_ratio else self.max_size
        # # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        # image = F.resize(image, size)
        # # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # # 如果是之前的版本需要使用PIL.Image.NEAREST
        # target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)

        ow, oh = image.size
        ratio = size / max(ow, oh)
        resize_w_h = [int(ow * ratio + 0.5), int(oh * ratio + 0.5)]
        # image = F.resize(image, [int(oh * ratio), int(ow * ratio)])
        image = image.resize(resize_w_h)
        target['landmark'] = {i: [j[0] * ratio, j[1] * ratio] for i, j in target['landmark'].items()}
        target['poly_mask'] = target['poly_mask'].resize(resize_w_h, resample=Image.NEAREST)
        target['transforms'].append('RandomResize')
        target['resize_ratio'] = ratio

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            target['poly_mask'] = np.ascontiguousarray(np.flip(target['poly_mask'], axis=[1]))
            h, w = image.shape[:2]
            target['landmark'] = {i: [w - j[0], j[1]] for i, j in target['landmark'].items()}
            target['transforms'].append('RandomHorizontalFlip')
            target['h_flip'] = True
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = np.ascontiguousarray(np.flip(image, axis=[0]))
            target['poly_mask'] = np.ascontiguousarray(np.flip(target['poly_mask'], axis=[0]))
            h, w = image.shape[:2]
            target['landmark'] = {i: [j[0], h - j[1]] for i, j in target['landmark'].items()}
            target['transforms'].append('RandomVerticalFlip')
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 未完成
        image = self.pad_if_smaller(image, self.size)
        target = self.pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

    def pad_if_smaller(self, img, size, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        if isinstance(img, torch.Tensor):
            img_size = img.shape[-2:]
        elif isinstance(img, Image.Image):
            img_size = img.size  # .size为Image里的方法
        else:
            raise '图像类型错误'
        min_size = min(img_size)
        if min_size < size:
            ow, oh = img_size
            padh = size - oh if oh < size else 0
            padw = size - ow if ow < size else 0
            img = F.pad(img, (0, 0, padw, padh), fill=fill)
        return img


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        target['show_img'] = image.copy()
        image = F.to_tensor(image)
        target['mask'] = torch.as_tensor(np.array(target['mask']), dtype=torch.float32)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RightCrop(object):
    def __init__(self, size=2 / 3):
        self.size = size

    def __call__(self, image, target):
        w, h = image.size
        image = F.crop(image, 0, 0, height=int(self.size * h), width=int(self.size * w))
        target = F.crop(target, 0, 0, height=int(self.size * h), width=int(self.size * w))
        return image, target


class BatchResize(object):
    def __init__(self, size):
        self.size = size
        self.stride = 32

    def __call__(self, image, target):
        max_size = self.max_by_axis([list(img.shape) for img in image])
        max_size[1] = int(math.ceil(max_size[1] / self.stride) * self.stride)
        max_size[2] = int(math.ceil(max_size[2] / self.stride) * self.stride)

        #
        img_batch_shape = [len(image)] + [3, max_size[1], max_size[2]]
        target_batch_shape = [len(image)] + [6, max_size[1], max_size[2]]
        # 创建shape为batch_shape且值全部为255的tensor
        batched_imgs = image[0].new_full(img_batch_shape, 255)
        batched_target = target[0].new_full(target_batch_shape, 255)
        for img, pad_img, t, pad_t in zip(image, batched_imgs, target, batched_target):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_t[: t.shape[0], : t.shape[1], : t.shape[2]].copy_(t)

        # image = F.resize(batched_imgs, [self.size,self.size])
        # target = F.resize(batched_target, [self.size,self.size])
        return image, target

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes


class MyCrop(object):  # 左右裁剪1/6 ,下裁剪1/3
    def __init__(self, left_size=1 / 6, right_size=1 / 6, bottom_size=1 / 3):
        self.left_size = left_size
        self.right_size = right_size
        self.bottom_size = bottom_size

    def __call__(self, img, target=None):
        img_w, img_h = img.size
        top = 0
        left = int(img_w * self.left_size)
        height = int(img_h * (1 - self.bottom_size))
        width = int(img_w * (1 - self.left_size - self.right_size))
        image = F.crop(img, top, left, height, width)
        if target is not None:
            target = F.crop(target, top, left, height, width)
            return image, target
        return image


class GetROI(object):
    def __init__(self, border_size=10):
        self.border_size = border_size

    def __call__(self, img, target):
        img_w, img_h = img.size
        mask = target['mask']
        landmark = target['landmark']
        # 训练和验证，求出roi box
        if target['roi_box'] is None:
            y, x = np.where(mask != 0)
            # 将landmark的值加入
            x, y = x.tolist(), y.tolist()
            x.extend([int(i[0]) for i in landmark.values()])
            y.extend([int(i[1]) for i in landmark.values()])
            left, right = min(x) - self.border_size, max(x) + self.border_size
            top, bottom = min(y) - self.border_size, max(y) + self.border_size
            left = left if left > 0 else 0
            right = right if right < img_w else img_w
            top = top if top > 0 else 0
            bottom = bottom if bottom < img_h else img_h
            height = bottom - top
            width = right - left
        # 测试集，使用detec module预测得到的roi box
        else:
            box = target['roi_box']
            left, top, right, bottom = box
            assert left >= 0 and top >= 0 and right <= img_w and bottom <= img_h
            height, width = bottom - top, right - left

        roi_img = F.crop(img, top, left, height, width)
        roi_mask = F.crop(mask, top, left, height, width)
        roi_landmark = {i: [j[0] - left, j[1] - top] for i, j in landmark.items()}
        roi_curve = {i: [[j[0] - left, j[1] - top] for j in target['curve'][i]] for i in target['curve']}
        return roi_img, roi_mask, roi_landmark, roi_curve, [left, top, right, bottom]


class MyPad(object):
    def __init__(self, size):
        if len(size) == 1:
            self.size_h = self.size_w = size[0]
        else:
            self.size_h, self.size_w = size[0], size[1]

    def __call__(self, img, target):
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            h, w = img.shape[-2:]
        w_pad, h_pad = self.size_w - w, self.size_h - h
        data_type = target['data_type']
        landmark = target['landmark']
        # if data_type == 'val':
        #     pad_size = [w_pad // 2, h_pad // 2, w_pad - w_pad // 2, h_pad - h_pad // 2]
        #     landmark = {i: [j[0] + w_pad // 2, j[1] + h_pad // 2] for i, j in landmark.items()}
        if data_type == 'val' or data_type == 'test':
            w_pad_l = 0
            h_pad_l = 0
        else:  # train 可以用各种填充方式
            random_ = np.random.random()
            if random_ < 0.5:
                w_pad_l = 0
                h_pad_l = 0
            elif random_ < 0.8:
                w_pad_l = np.random.randint(0, w_pad + 1) if w_pad >= 0 else np.random.randint(w_pad, 1)
                h_pad_l = np.random.randint(0, h_pad + 1) if h_pad >= 0 else np.random.randint(h_pad, 1)
            else:
                w_pad_l, h_pad_l = int(w_pad / 2), int(h_pad / 2)
        pad_size = [w_pad_l, h_pad_l, w_pad - w_pad_l, h_pad - h_pad_l]
        landmark = {i: [j[0] + w_pad_l, j[1] + h_pad_l] for i, j in landmark.items()}

        img = F.pad(img, pad_size, fill=0)
        target['landmark'] = landmark
        target['mask'] = F.pad(target['mask'], pad_size, fill=255)
        return img, target


class RandomRotation(object):
    def __init__(self, degree, rotate_ratio=1, expand_ratio=1):
        self.degree = degree
        self.rotate_ratio = rotate_ratio
        self.expand_ratio = expand_ratio

    def __call__(self, img, target):
        if np.random.random() > self.rotate_ratio:
            return img, target
        degree = random.randint(-self.degree, self.degree)
        expand_ = True if np.random.random() > self.expand_ratio else False
        img = F.rotate(img, degree, expand=expand_, fill=[0, 0, 0])
        target['mask'] = F.rotate(target['mask'], degree, expand=expand_, fill=[255])

        return img, target

    def rotate_point(self, point1, point2, angle, height):
        """
        点point1绕点point2旋转angle后的点
        ======================================
        在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：
        x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
        y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
        ======================================
        将图像坐标(x,y)转换到平面坐标(x`,y`)：
        x`=x
        y`=height-y
        :param point1:
        :param point2: base point (基点)
        :param angle: 旋转角度，正：表示逆时针，负：表示顺时针
        :param height:
        :return:
        """
        x1, y1 = point1
        x2, y2 = point2
        # 将图像坐标转换到平面坐标
        y1 = height - y1
        y2 = height - y2
        x = (x1 - x2) * np.cos(np.pi / 180.0 * angle) - (y1 - y2) * np.sin(np.pi / 180.0 * angle) + x2
        y = (x1 - x2) * np.sin(np.pi / 180.0 * angle) + (y1 - y2) * np.cos(np.pi / 180.0 * angle) + y2
        # 将平面坐标转换到图像坐标
        y = height - y
        return (x, y)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img, mask):
        brightness = random.uniform(max(0, 1 - self.brightness),
                                    1 + self.brightness)  # 将图像的亮度随机变化为原图亮度的（1−bright）∼（1+bright）
        contrast = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        img = F.adjust_brightness(img, brightness)
        img = F.adjust_contrast(img, contrast)
        img = F.adjust_saturation(img, saturation)
        return img, mask


class PepperNoise(object):
    def __init__(self, snr, p):
        self.snr = snr
        self.p = p

    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask_ = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask_ = np.repeat(mask_, c, axis=2)
            img_[mask_ == 1] = 255  # 盐噪声
            img_[mask_ == 2] = 0  # 椒噪声
            img = F.to_pil_image(img_)
        return img, mask


class Resize(object):
    def __init__(self, size):
        if len(size) == 2:
            size = size[0]
        self.size = size[0]

    def __call__(self, image, target):
        ow, oh = image.size
        ratio = self.size / max(ow, oh)
        resize_w_h = [int(ow * ratio + 0.5), int(oh * ratio + 0.5)]
        # image = F.resize(image, [int(oh * ratio), int(ow * ratio)])
        image = image.resize(resize_w_h)
        target['landmark'] = {i: [j[0] * ratio, j[1] * ratio] for i, j in target['landmark'].items()}
        target['poly_mask'] = target['poly_mask'].resize(resize_w_h, resample=Image.NEAREST)
        target['transforms'].append('Resize')
        target['resize_ratio'] = ratio

        return image, target


class GenerateMask(object):
    def __init__(self, var=40, task='landmark', max_value=8):
        self.var = var
        self.max_value = max_value
        self.task = task

    def __call__(self, img, target):
        # generate mask according to the task type
        num_c = 2 if self.task == 'landmark' else 3 if self.task == 'poly' else 5
        mask = torch.zeros((num_c, *img.shape[:2]), dtype=torch.float)
        # generate landmark mask
        if self.task in ['landmark', 'all']:
            landmark_mask = torch.zeros((2, *img.shape[:2]), dtype=torch.float)
            # 生成landmark, landmark的误差在int()处
            landmark = {i: [int(target['landmark'][i][0] + 0.5), int(target['landmark'][i][1] + 0.5)] for i in
                        target['landmark']}
            # 根据landmark 绘制高斯热图 （进行点分割）
            for label in landmark:
                point = landmark[label]
                temp_heatmap = self.__make_2d_heatmap(point, img.shape[:2], var=self.var, max_value=self.max_value)
                landmark_mask[label - 5] = temp_heatmap
            if target['h_flip']:  # 保持左右
                landmark_mask[[0, 1], ::] = landmark_mask[[1, 0], ::]
                target['landmark'][5], target['landmark'][6] = target['landmark'][6], target['landmark'][5]
            mask[:2, :, :] = landmark_mask
        # generate poly mask, to avoid anno mistakes, check the annotation data
        if self.task in ['poly', 'all']:
            poly_mask = np.eye(3)[target['poly_mask']].transpose(2, 0, 1)
            if target['h_flip']:   # 保持左右
                poly_mask[[1, 2], ::] = poly_mask[[2, 1], ::]
            poly_mask = torch.from_numpy(poly_mask)
            mask[-3:, :, :] = poly_mask
            # mask[-3][poly_mask[0]+poly_mask[1] == 0] = 1
        target['mask'] = mask
        target['transforms'].append('GenerateHeatmap')
        return img, target

    def __make_2d_heatmap(self, landmark, size, max_value=None, var=5.0):
        """
        生成一个size大小，中心在landmark处的热图
        :param max_value: 热图中心的最大值
        :param var: 生成热图的方差 （不是标准差）
        """
        height, width = size
        landmark = (landmark[1], landmark[0])
        x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing="ij")  # 一个网格有横纵两个坐标
        p = torch.stack([x, y], dim=2)
        from math import pi, sqrt
        inner_factor = -1 / (2 * var)
        outer_factor = 1 / sqrt(var * (2 * pi))
        mean = torch.as_tensor(landmark)
        heatmap = (p - mean).pow(2).sum(dim=-1)
        heatmap = torch.exp(heatmap * inner_factor)

        # heatmap[heatmap == 1] = 5
        # 将heatmap的最大值进行缩放
        if max_value is not None:
            heatmap = heatmap * max_value
        return heatmap


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


class AffineTransform(object):
    """scale+rotation"""
    # 仿射变换最重要的是计算变换矩阵
    # opencv可以根据三个点变换前后的对应关系自动求解：affine_matrix = cv2.getAffineTransform(src, dst)
    # 然后使用cv2.warpAffine()进行仿射变换
    def __init__(self,
                 input_size: Tuple[int, int] = (192, 256),  # 输入网络的图片大小 （h*w）
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35) 将图片放大或缩小的比例
                 rotation: Tuple[int, int] = None,   # e.g. (-45, 45)  将图片旋转的角度
                 resize_low_high=(1, 1),  # 对输入网络的大小resize，比例从low-high中随机采样
                 heatmap_shrink_rate: int = 1,  # heatmap缩小尺寸，如hrnet会将预测的heatmap缩小
                 stretch: bool = False,  # resize的过程中，拉伸图像不padding，保持长宽比后padding会引入噪声
                 ):
        self.scale = scale
        self.rotation = rotation
        self.input_size = input_size
        self.heatmap_shrink_rate = heatmap_shrink_rate
        self.resize_low_high = resize_low_high
        self.stretch = stretch

    def __call__(self, img, target):
        resize_ratio = np.random.uniform(*self.resize_low_high)
        src_xmax, src_xmin, src_ymax, src_ymin = img.size[0], 0, img.size[1], 0
        """ resize 选项 """
        # 1. 直接拉伸到self.input_size
        # 2. 如果为训练集， 拉伸过程中缩小，后续使用padding
        # 3. 保持长宽比，resize 过程中缩小，后续使用padding
        if self.stretch:   # Fixme 拉伸时，如果resize缩小后再padding无法训练，dice loss会变为0
            if np.random.uniform(0, 1) < 0.5 and target['data_type'] in ['train']:
                input_size = [int(i * resize_ratio+0.5) for i in self.input_size]
            else:
                input_size = self.input_size
        else:
            # 将长边(w或h), resize到resize_ratio * 256  todo 不使用256大小
            # 保持长宽比时，一定需要padding，所以此时可以使用缩小
            input_size = [int(src_ymax/src_xmax*(256*resize_ratio)), int(256*resize_ratio)] \
                if src_xmax > src_ymax \
                else [int(256*resize_ratio), int(src_xmax/src_ymax*(256*resize_ratio))]
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        # 注意放射变换src和dst中坐标为:w,h, numpy中为h,w
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        dst_center = np.array([(input_size[1] - 1) / 2, (input_size[0] - 1) / 2])
        dst_p2 = np.array([(input_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([input_size[1] - 1, (input_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            # scale < 1, 图像放大，区域内显示的图形内容变少， scale > 1，图像缩小，区域内显示的图像内容变多（填充黑边）
            scale = random.uniform(*self.scale)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        if self.rotation is not None:
            angle = random.randint(*self.rotation)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
        dst /= self.heatmap_shrink_rate  # 网络预测的heatmap尺寸是输入图像的1/4
        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        resize_img = cv2.warpAffine(np.array(img),
                                    trans,
                                    tuple(input_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)
        target['poly_mask'] = cv2.warpAffine(np.array(target['poly_mask']), trans, tuple(input_size[::-1]),
                                             flags=cv2.INTER_NEAREST)

        affine_landmark = np.array([target['landmark'][5], target['landmark'][6]])
        affine_landmark = affine_points(affine_landmark, trans) / self.heatmap_shrink_rate
        target['landmark'][5] = affine_landmark[0]
        target['landmark'][6] = affine_landmark[1]

        # import matplotlib.pyplot as plt
        # from draw_utils import draw_keypoints
        # resize_img = draw_keypoints(resize_img, target["keypoints"])
        # plt.imshow(resize_img)
        # plt.show()

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        target['transforms'].append('AffineTransform')
        return resize_img, target
