import os
import json
import math

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from PIL import Image
from scipy.ndimage import label
from data_utils.visualize import plot_result

from dataSet import get_name_data
from train_multi_GPU import get_transform
from train_utils.init_model_utils import create_model
from train_utils.distributed_utils import get_default_device
from train_utils.dice_coefficient_loss import multiclass_dice_coeff
from torch.nn.functional import softmax


def main():
    # init basic setting
    data_root = '../datas/IRD/COCO_style'
    model_path = 'model/20240224/poly/unet_seg_od_bc32_nlf_nstretch_4-all_0.912'
    device = get_default_device()
    print("using {} device.".format(device))
    init_img = torch.zeros((1, 3, 256, 256), device=device)

    # load model config
    with open(os.path.join(model_path, 'config.json')) as reader:
        model_config = json.load(reader)
    task = model_config['task']
    num_classes = 2 if task == 'landmark' else 3 if task == 'poly' else 5
    model_weight_name = 'best_model.pth' if task in ['landmark', 'all'] else 'best_dice_model.pth'
    position_type = model_config['position_type']
    model_name = model_config['model_name'] if model_config.get('model_name') else 'unet'
    model_base_c = model_config['base_c'] if model_config.get('base_c') else model_config['unet_bc']
    intput_size = model_config['input_size'] if model_config.get('input_size') \
        else [model_config['base_size'], model_config['base_size']]  # 输入模型的图像尺寸

    # init model
    model = create_model(num_classes=num_classes, base_c=model_base_c, model_name=model_name)
    model.load_state_dict(torch.load(os.path.join(model_path, model_weight_name), map_location=device)['model'])
    model.to(device).eval()
    model(init_img)

    # init dataset
    with open('data_utils/data.json', 'r') as reader:
        data_list = json.load(reader)[position_type]
        mean = data_list['train_info']['mean']
        std = data_list['train_info']['std']

    # init save result
    save_root = model_path.replace('model', 'result')
    restore_ori_size = True
    for i in ['result', 'heatmap', 'ori_img']:
        os.makedirs(os.path.join(save_root, i), exist_ok=True)
    result = {'name': [], 'left_mse': [], 'right_mse': []} if task == 'landmark' else None
    result = {'name': [], 'left_dice': [], 'right_dice': [], 'dice': []} if task == 'poly' else result
    result = {'name': [], 'left_mse': [], 'right_mse': [], 'left_dice': [], 'right_dice': [], 'dice': []} \
        if task == 'all' else result

    # begin to predict
    for i, name in enumerate(data_list['val']):
        print(name)
        result['name'].append(name)

        # get val data img and target
        ori_img, ori_landmark, ori_mask = get_name_data(data_root, name)
        transforms = get_transform(train=False, input_size=intput_size, task=task, var=model_config['var'],
                                   max_value=model_config['max_value'], mean=mean, std=std)
        input_img, target = transforms(ori_img, {'landmark': ori_landmark, 'poly_mask': ori_mask, 'data_type': 'val',
                                                 'transforms': [], 'h_flip': False})
        input_img = input_img.unsqueeze(0).to(device)
        resize_h, resize_w = target['show_img'].shape[:2]
        show_img = target['show_img'] if not restore_ori_size else np.array(ori_img)

        # run model, get output and remove padding
        output = model(input_img).to('cpu').detach()[0]
        output = np.array(output)[:, :resize_h, :resize_w]

        # save the output heatmap result
        for pre_ind, pre_channel in enumerate(output):
            plt.imsave(os.path.join(save_root, 'heatmap', name + f'_{str(pre_ind)}.png'), pre_channel)

        # get pre result
        landmark_pre, mask_pre = generate_pre_target(output, task, restore_ori_size, ori_img.size)
        landmark_gt, mask_gt = {}, {}

        # analyse result
        if task in ['landmark', 'all']:
            landmark_gt = target['landmark'] if not restore_ori_size else ori_landmark
            # 计算mse用的与restore_ori_size匹配的landmark
            for i in landmark_pre:
                left_right = 'left' if i == 5 else 'right'
                point_pre = landmark_pre[i]
                point_gt = landmark_gt[i]
                result[left_right + '_mse'].append(
                    math.sqrt(math.pow(point_pre[0] - point_gt[0], 2) + math.pow(point_pre[1] - point_gt[1], 2)))

        if task in ['poly', 'all']:
            mask_gt = np.array(target['mask'])[:, :resize_h, :resize_w] if not restore_ori_size \
                else np.eye(3)[ori_mask].transpose(2, 0, 1)
            # 计算dice用的测试尺寸大小的结果
            dice_gt = target['mask'][-3:, :resize_h, :resize_w].unsqueeze(0)
            dices = np.array(multiclass_dice_coeff(softmax(torch.from_numpy(output[-3:, :, :]), 0).unsqueeze(0), dice_gt))
            result['dice'].append(dices.mean())
            result['left_dice'].append(dices[1])
            result['right_dice'].append(dices[2])

        # plot pre result and save it
        plot_result(show_img, target={'landmark': landmark_gt, 'mask': mask_gt},
                    pre_target={'landmark': landmark_pre, 'mask': mask_pre}, task=task, show=False,
                    save_path=os.path.join(save_root, 'result'), title=name + '_os' if restore_ori_size else name)
        # copy original img for analyse
        shutil.copyfile(os.path.join(data_root, 'images', name+'.png'), os.path.join(save_root, 'ori_img', name+'.png'))

    df = pd.DataFrame(result)
    df.to_excel(os.path.join(save_root, f'{os.path.basename(model_path)}.xlsx'), index=False)


def generate_pre_target(output, task='landmark', restore_ori_size=False, ori_size=(0, 0)):
    pre_landmark = {}
    pre_mask = np.zeros((2, *output.shape[-2:])) if not restore_ori_size else np.zeros((2, *ori_size[::-1]))
    resize_output = output if not restore_ori_size else np.zeros((output.shape[0], *ori_size[::-1]))

    # restore to ori size
    if restore_ori_size:
        for ind, pre in enumerate(output):
            resize_output[ind] = cv2.resize(pre, ori_size)

    if task in ['landmark', 'all']:
        for i, pre in enumerate(resize_output[:2]):
            y, x = np.where(pre == pre.max())
            pre_landmark[i + 5] = [x[0], y[0]]

    if task in ['poly', 'all']:
        pre_ = np.argmax(resize_output[-3:], axis=0)
        for ind in range(1, 3):
            label_ind = label(pre_ == ind)
            # todo 只取了最大的分割区域，是否有利？？？
            if label_ind[1] > 1:
                pre_mask[ind-1] = (label_ind[0] == np.argmax(np.bincount(label_ind[0].flatten())[1:]) + 1)
            else:
                pre_mask[ind-1] = label_ind[0]

    return pre_landmark, pre_mask


if __name__ == '__main__':
    main()
