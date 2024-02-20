import os
import json
import math

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataSet import get_name_data
from train_multi_GPU import create_model, get_transform
from train_utils.distributed_utils import get_default_device


def main():
    # init basic setting
    data_root = '../datas/IRD/COCO_style'
    model_path = 'model/20240218/landmark/unet_keypoint_1_4-all_var40_6.551'
    model_weight_name = 'best_model.pth'
    device = get_default_device()
    print("using {} device.".format(device))
    init_img = torch.zeros((1, 3, 256, 256), device=device)

    # load model config
    with open(os.path.join(model_path, 'config.json')) as reader:
        model_config = json.load(reader)
    task = model_config['task']
    num_classes = 2 if task in ['landmark', 'poly'] else 4
    position_type = model_config['position_type']
    model_name = model_config['model_name'] if model_config.get('model_name') else 'unet'
    model_base_c = model_config['base_c'] if model_config.get('base_c') else model_config['unet_bc']
    base_size = model_config['base_size']  # 输入模型的图像尺寸

    # init model
    model = create_model(num_classes=num_classes, base_c=model_base_c, model=model_name)
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
    for name in ['result', 'heatmap']:
        os.makedirs(os.path.join(save_root, name), exist_ok=True)
    result = {'name': [], 'left_mse': [], 'right_mse': []} if task == 'landmark' else None
    result = {'name': [], 'left_dice': [], 'right_dice': []} if task == 'poly' else result
    result = {'name': [], 'left_mse': [], 'right_mse': [], 'left_dice': [],
              'right_dice': []} if task == 'all' else result

    # begin to predict
    for i, name in enumerate(data_list['val']):
        print(name)
        result['name'].append(name)

        # get val data img and target
        ori_img, ori_landmark, ori_mask = get_name_data(data_root, name)
        transforms = get_transform(train=False, base_size=base_size, task=task, var=model_config['var'],
                                   max_value=model_config['max_value'], mean=mean, std=std)
        input_img, target = transforms(ori_img, {'landmark': ori_landmark, 'poly_mask': ori_mask, 'data_type': 'val',
                                                 'transforms': [], 'h_flip': False})
        input_img = input_img.unsqueeze(0).to(device)
        resize_w, resize_h = target['show_img'].size
        show_img = np.array(target['show_img']) if not restore_ori_size else np.array(ori_img)

        # run model, get output and remove padding
        output = model(input_img).to('cpu').detach()[0]
        output = np.array(output)[:, :resize_h, :resize_w]

        # analyse result
        if task in ['landmark', 'all']:
            landmark_gt = target['landmark'] if not restore_ori_size else ori_landmark
            landmark_pre = {ind: [0, 0] for ind in landmark_gt}
            for i, pre in enumerate(output[:2]):
                pre = pre if not restore_ori_size else cv2.resize(pre, show_img.shape[:2][::-1])
                left_right = 'left' if i == 0 else 'right'
                y, x = np.where(pre == pre.max())
                landmark_pre[i + 5] = [x[0], y[0]]
                point = landmark_gt[i + 5]  # 使用没有经过int处理的坐标计算结果，与训练时统一
                result[left_right + '_mse'].append(
                    math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)))
                # save heatmap result
                plt.imsave(os.path.join(save_root, 'heatmap', name + f'_{left_right}.png'), np.array(pre))

            landmark_gt = {ind: [int(item[0] + 0.5), int(item[1] + 0.5)] for ind, item in landmark_gt.items()}
            cv2.circle(show_img, landmark_gt[5], 1, [255, 0, 0], -1)
            cv2.circle(show_img, landmark_gt[6], 1, [255, 0, 0], -1)
            cv2.circle(show_img, landmark_pre[5], 1, [0, 255, 0], -1)
            cv2.circle(show_img, landmark_pre[6], 1, [0, 255, 0], -1)
            cv2.putText(show_img, f'left_mse: {round(result["left_mse"][-1], 2)}pix', [20, show_img.shape[0] - 35],
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(show_img, f'right_mse: {round(result["right_mse"][-1], 2)}pix', [20, show_img.shape[0] - 15],
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
            show_img = Image.fromarray(show_img)
            show_img.save(os.path.join(save_root, 'result', name + '.png'))

        if task in ['poly', 'all']:
            pass
    df = pd.DataFrame(result)
    df.to_excel(os.path.join(save_root, 'mse_mm.xlsx'), index=False)


if __name__ == '__main__':
    main()
