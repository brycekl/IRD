import os
import json
import math

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataSet import IRDDataset
from train_multi_GPU import create_model, get_transform
from train_utils.distributed_utils import get_default_device


def main():
    # init basic setting
    model_path = 'model/240529/landmark/14_14_var40_3.989'
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
    result = {'name': [], 'left_mse': [], 'right_mse': []} if task == 'landmark' else None
    result = {'name': [], 'left_dice': [], 'right_dice': []} if task == 'poly' else result
    result = {'name': [], 'left_mse': [], 'right_mse': [], 'left_dice': [], 'right_dice': []} if task == 'all' else result

    # begin to predict
    for img, target in val_data_loader:
        name = target['img_name'][0]
        print(name)
        img = img.to(device)
        output = model(img).to('cpu').detach()
        show_img = np.array(target['show_img'][0])
        result['name'].append(name)
        if task in ['landmark', 'all']:
            landmark_gt = {ind: [int(item[0] + 0.5), int(item[1] + 0.5)] for ind, item in target['landmark'][0].items()}
            landmark_pre = {ind: [0, 0] for ind in landmark_gt}
            for i, pre in enumerate(output[0][:2]):
                left_right = 'left' if i == 0 else 'right'
                y, x = np.where(pre == pre.max())
                landmark_pre[i+5] = [x[0], y[0]]
                point = target['landmark'][0][i + 5]  # label=i+8
                result[left_right+'_mse'].append(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)))
                # save heatmap result
                pre_array = np.array(pre)
                pre_array = (pre_array - pre_array.min()) / (pre_array.max() - pre_array.min()) * 255
                pre_image = Image.fromarray(pre_array.astype(np.uint8))
                pre_image.save(os.path.join(save_root, 'heatmap', name + f'_{left_right}.png'))

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
