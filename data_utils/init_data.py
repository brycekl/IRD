import os
import json
import random
from PIL import Image
import numpy as np


posture_label = {'9': 'PP'}
position_label = {'12': 'B3', '13': 'U3', '14': 'U5', '15': 'UE'}
random.seed(2024)


def generate_json():
    root = '../../datas/IRD/COCO_style/jsons'
    save_json = {'12': [], '13': [], '14': [], '15': [], '4-all': [], 'all': []}
    json_files = [item.split('.json')[0] for item in os.listdir(root)]
    for json_file in json_files:
        json_path = os.path.join(root, json_file + '.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_posture = [str(item['Label']) for item in data['Models']['FrameLabelModel']['ChildFrameLabel']]
        inter = set(list(position_label.keys()) + ['9']).intersection(data_posture)
        if '9' in inter:
            for item in data_posture:
                if item in list(position_label.keys()):
                    save_json[item].append(json_file)
                    save_json['4-all'].append(json_file)
        save_json['all'].append(json_file)

    final_data = {i: {} for i in save_json.keys()}
    for i, item in save_json.items():
        # FIXME all 数据类型不应该将其他的数据划入其中
        sampled_list_1 = random.sample(item, int(len(item) * 0.8))
        sampled_list_2 = [item for item in item if item not in sampled_list_1]
        train_info = compute_mean_std(os.path.join(os.path.dirname(root), 'images'), sampled_list_1)
        final_data[i]['train'] = sampled_list_1
        final_data[i]['val'] = sampled_list_2
        final_data[i]['train_info'] = train_info
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f)


def compute_mean_std(img_root, data_list):
    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    height, width = [], []
    height_max, height_min = 0, 10000
    width_max, width_min = 0, 10000
    for item in data_list:
        img_path = os.path.join(img_root, os.path.basename(item).split('.json')[0] + '.jpg')
        img = Image.open(img_path)
        if img_channels == 1: img.convert('L')
        elif img_channels == 3: img.convert('RGB')
        w, h = img.size
        height.append(h)
        width.append(w)
        width_max = max(width_max, w)
        width_min = min(width_min, w)
        height_max = max(height_max, h)
        height_min = min(height_min, h)

        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min())
        assert img.max() == 1. and img.min() == 0., item
        # gray image
        cumulative_mean += img.mean()
        cumulative_std += img.std()
        # rgb image
        # img = img.reshape(-1, 3)
        # cumulative_mean += img.mean(axis=0)
        # cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(data_list)
    std = cumulative_std / len(data_list)
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f'average height : {np.mean(height)},    height std : {np.std(height)}')
    print(f'average width : {np.mean(width)},    width std : {np.std(width)}')
    print(f'max height : {height_max}   min height : {height_min}')
    print(f'max width : {width_max}   min width : {width_min}')
    return {'mean': list(mean), 'std': list(std), 'max_h': height_max, 'max_w': width_max,
            'min_h': height_min, 'min_w': width_min, 'mean_h': np.mean(height), 'mean_w': np.mean(width),
            'std_h': np.std(height), 'std_w': np.std(width)}


if __name__ == '__main__':
    generate_json()
