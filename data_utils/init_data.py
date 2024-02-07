import os
import json


def generate_json():
    root = '../../datas/IRD/COCO_style/jsons'
    json_files = os.listdir(root)
    for json_file in json_files:
        json_path = os.path.join(root, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)



if __name__ == '__main__':
    generate_json()