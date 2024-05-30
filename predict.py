import os
import json
import math

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataSet import IRDDataset, get_name_data
from train_multi_GPU import create_model, get_transform
from train_utils.distributed_utils import get_default_device
from data_utils.visualize import plot_result


def main():
    # init basic setting
    model_path = 'model/240528-clahe/poly/14_tall_14_0.904'
    model_weight_name = 'best_model.pth'
    device = get_default_device()
    print("using {} device.".format(device))
    init_img = torch.zeros((1, 3, 256, 256), device=device)

    # load model config
    with open(os.path.join(model_path, 'config.json')) as reader:
        config = json.load(reader)
    task = config['task']
    num_classes = 2 if task == 'landmark' else 3 if task == 'poly' else 5
    position_type = config['position_type']
    model_name = config['model_name'] if config.get('model_name') else 'unet'
    model_base_c = config['base_c'] if config.get('base_c') else config['unet_bc']

    # init model
    model = create_model(num_classes=num_classes, base_c=model_base_c, model=model_name)
    model.load_state_dict(torch.load(os.path.join(model_path, model_weight_name), map_location=device)['model'])
    model.to(device).eval()
    model(init_img)

    # init dataset
    with open('data_utils/data_info.json', 'r') as reader:
        json_list = json.load(reader)[position_type]
        mean = json_list['train_info']['mean']
        std = json_list['train_info']['std']
    val_dataset = IRDDataset(data_type='val', position_type=position_type, task=task, clahe=config['clahe'],
                             transforms=get_transform(train=False, input_size=config['input_size'], task=task,
                                                      var=config['var'], max_value=config['max_value'],
                                                      mean=mean, std=std, stretch=config['stretch']))
    test_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=test_sampler, num_workers=1,
                                                  collate_fn=val_dataset.collate_fn)

    # init save result
    save_root = model_path.replace('model', 'result')
    for name in ['result', 'heatmap']:
        os.makedirs(os.path.join(save_root, name), exist_ok=True)
    if task == 'landmark':
        result = {'name': [], 'left_mse': [], 'right_mse': [], 'l_dis_gt': [], 'l_dis_pre': []}
    elif task == 'poly':
        result = {'name': [], 'left_dice': [], 'right_dice': [], 'p_dis_gt': [], 'p_dis_pre': []}
    else:
        result = {'name': [], 'left_mse': [], 'right_mse': [], 'left_dice': [], 'right_dice': [],
                  'l_dis_gt': [], 'l_dis_pre': [], 'p_dis_gt': [], 'p_dis_pre': []}
    vis_gt_target, pre_target = {}, {}   # for visualization
    with open('./data_utils/spacing.json') as reader:
        spacing = json.load(reader)

    # begin to predict
    for img, target in val_data_loader:
        name = target['img_name'][0]
        spa_name = 1 if name.split('__')[0] not in spacing else spacing[name.split('__')[0]] * 10  # mm
        print(name)
        img = img.to(device)
        output = model(img).to('cpu').detach().numpy()[0]
        show_img, landmark_gt, poly_gt = get_name_data('../datas/IRD/COCO_style', name)
        show_img = np.stack([show_img] * 3, axis=2)
        result['name'].append(name)

        if task in ['landmark', 'all']:
            output_landmark = cv2.warpAffine(np.transpose(output[:2], (1, 2, 0)), target['reverse_trans'][0],
                                             show_img.shape[:2][::-1], flags=cv2.INTER_LINEAR)
            landmark_pre = {ind: [0, 0] for ind in landmark_gt}
            for i in range(2):
                pre = output_landmark[:, :, i]
                left_right = 'left' if i == 0 else 'right'
                y, x = np.where(pre == pre.max())
                landmark_pre[i+5] = [x[0], y[0]]
                point = landmark_gt[i + 5]  # label=i+8
                result[left_right+'_mse'].append(spa_name * math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)))
                # save heatmap result
                pre_array = np.array(pre)
                pre_array = (pre_array - pre_array.min()) / (pre_array.max() - pre_array.min()) * 255
                pre_image = Image.fromarray(pre_array.astype(np.uint8))
                pre_image.save(os.path.join(save_root, 'heatmap', name + f'_{left_right}.png'))

            landmark_gt = {i: [int(j[0] + 0.5), int(j[1] + 0.5)] for i, j in landmark_gt.items()}
            vis_mask = np.zeros((2, *show_img.shape[:2]))
            vis_mask[0][poly_gt == 1] = 1
            vis_mask[1][poly_gt == 2] = 1
            vis_gt_target['landmark'] = landmark_gt
            vis_gt_target['mask'] = vis_mask
            pre_target['landmark'] = landmark_pre
            pre_target['mse'] = {5: result["left_mse"][-1], 6: result["right_mse"][-1]}

            # Calculate the rectus abdominis distance
            dis_gt = spa_name * math.sqrt(math.pow(landmark_gt[5][0] - landmark_gt[6][0], 2) +
                                          math.pow(landmark_gt[5][1] - landmark_gt[6][1], 2))
            dis_pre = spa_name * math.sqrt(math.pow(landmark_pre[5][0] - landmark_pre[6][0], 2) +
                                           math.pow(landmark_pre[5][1] - landmark_pre[6][1], 2))
            result['l_dis_gt'].append(dis_gt)
            result['l_dis_pre'].append(dis_pre)
            pre_target['l_dis'] = {'l_dis_gt': dis_gt, 'l_dis_pre': dis_pre}

        if task in ['poly', 'all']:
            output_poly = cv2.warpAffine(np.transpose(output[-3:], (1, 2, 0)), target['reverse_trans'][0],
                                         show_img.shape[:2][::-1], flags=cv2.INTER_NEAREST)
            output_poly = np.argmax(output_poly, axis=2)
            pre_mask = np.zeros((2, *show_img.shape[:2]))
            gt_mask = np.zeros((2, *show_img.shape[:2]))
            keypoint = {'pre': {5: [], 6: []}, 'gt': {5: [], 6: []}}  # the keypoint of poly
            for i in range(2):
                pre_ = output_poly == (i+1)
                gt_ = poly_gt == (i+1)
                intersection = np.logical_and(pre_, gt_)
                dice = 2 * intersection.sum() / (pre_.sum() + gt_.sum())
                left_right = 'left' if i == 0 else 'right'
                result[f'{left_right}_dice'].append(dice)
                pre_mask[i] = pre_
                gt_mask[i] = gt_
                # get keypoint of poly
                for type_, data in {'pre': pre_, 'gt': gt_}.items():
                    y, x = np.where(data)
                    if i == 0:
                        point = [x.max(), int(np.median(y[np.where(x == x.max())]))]
                    else:
                        point = [x.min(), int(np.median(y[np.where(x == x.min())]))]
                    keypoint[type_][i+5] = point

            pre_target['mask'] = pre_mask
            vis_gt_target['mask'] = gt_mask

            # Calculate the rectus abdominis distance
            dis_gt = spa_name * math.sqrt(math.pow(keypoint['gt'][5][0] - keypoint['gt'][6][0], 2) +
                                          math.pow(keypoint['gt'][5][1] - keypoint['gt'][6][1], 2))
            dis_pre = spa_name * math.sqrt(math.pow(keypoint['pre'][5][0] - keypoint['pre'][6][0], 2) +
                                           math.pow(keypoint['pre'][5][1] - keypoint['pre'][6][1], 2))
            result['p_dis_gt'].append(dis_gt)
            result['p_dis_pre'].append(dis_pre)
            pre_target['keypoint'] = keypoint
            pre_target['p_dis'] = {'p_dis_gt': dis_gt, 'p_dis_pre': dis_pre}

        plot_result(show_img, vis_gt_target, pre_target, task=config['task'], save_path=f'{save_root}/result', title=name)
    df = pd.DataFrame(result)
    df.to_excel(os.path.join(save_root, 'result.xlsx'), index=False)


if __name__ == '__main__':
    main()
