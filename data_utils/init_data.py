import os
import json
import random
import numpy as np


posture_label = {9: 'PP'}
position_label = {12: 'B3', 13: 'U3', 14: 'U5', 15: 'UE'}



def generate_json():
    root = '../../datas/IRD/COCO_style/jsons'
    save_json = {12: [], 13: [], 14: [], 15: [], '4-all': [], 'all': []}
    json_files = os.listdir(root)
    for json_file in json_files:
        json_path = os.path.join(root, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_posture = [item['Label'] for item in data['Models']['FrameLabelModel']['ChildFrameLabel']]
        inter = set(list(position_label.keys()) + [9]).intersection(data_posture)
        if 9 in inter:
            for item in data_posture:
                if item in list(position_label.keys()):
                    save_json[item].append(json_file)
                    save_json['4-all'].append(json_file)
        save_json['all'].append(json_file)

    final_data = {i: {} for i in save_json.keys()}
    for i, item in save_json.items():
        sampled_list_1 = random.sample(item, int(len(item) * 0.8))
        sampled_list_2 = [item for item in item if item not in sampled_list_1]
        final_data[i]['train'] = sampled_list_1
        final_data[i]['val'] = sampled_list_2
    with open('./data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f)


if __name__ == '__main__':
    generate_json()