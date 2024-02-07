import os
import json
import numpy as np
import random


def save_coco_json_dataset(images, annotations, save_path):
    json_data = {'images': [], 'annotations': []}

    # header information
    json_data['info'] = {'description': 'This is 1.0 version of the inter recti distance (IRD) MS COCO dataset.'
                                        'One picture corresponds to two bounding boxes.'
                                        'Used to train up bottom model.',
                         'url': None,
                         'version': '1.0',
                         'year': 2023,
                         'contributor': 'KaiLiu'}
    # init category info
    json_data['categories'] = [{'id': 1, 'keypionts': ['landmark'], 'name': 'abd_muscles', 'supercategory': 'infant'}]
    for image in images:
        json_data['images'].append(image)
    for anno in annotations:
        json_data['annotations'].extend(anno)
    # 写入json文件
    with open(save_path, 'w') as f:
        json.dump(json_data, f)


def collect_jsons(root):
    jsons = os.listdir(root)
    images, annotations = [], []
    for ind, json_name in enumerate(jsons):
        json_path = os.path.join(root, json_name)
        json_data = json.load(open(json_path))
        # 解析json文件得到边界框，关键点，分割区域，和体态信息
        bbox, keypoints, segmentations, titai, area = parse_json_data(json_data)

        image_info = {'id': ind,
                      'pair_version': json_data['FileInfo']['Version'],
                      'data_captured': json_data['FileInfo']['Date'],
                      'file_name': json_data['FileInfo']['Name'],
                      'height': json_data['FileInfo']['Height'],
                      'width': json_data['FileInfo']['Width'],
                      'posture': titai['posture'],
                      'position': titai['position']}
        images.append(image_info)

        anno_infos = []
        for category_id, category in enumerate(['left', 'right']):
            anno_info = {'id': 2*ind + category_id,
                         'image_id': ind,
                         'category_id': 1,
                         'bbox': bbox[category],
                         'area': area[category],
                         'iscrowd': 0,
                         'keypoints': keypoints[category],
                         'num_keypoints': 1,
                         'segmentation': [segmentations[category]]}
            anno_infos.append(anno_info)
        annotations.append(anno_infos)
    return images, annotations


def parse_json_data(json_data):
    bbox, keypoints, segmentations, titai, area = {}, {}, {}, {}, {}

    # 体态信息
    posture_label = {9: 'PP', 10: 'SP', 11: 'ST'}
    position_label = {12: 'B3', 13: 'U3', 14: 'U5', 15: 'UE'}
    for frame_label in json_data['Models']['FrameLabelModel']['ChildFrameLabel']:
        if frame_label['Label'] in posture_label:
            titai['posture'] = posture_label[frame_label['Label']]
        else:
            titai['position'] = position_label[frame_label['Label']]

    # 关键点信息
    k_label = {5: 'left', 6: 'right'}
    for keypoint_info in json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']:
        # todo : keypoints 第三个维度的含义, 1: 标注但没在标注的segmentation里，2：标注在标注的segmentation里
        keypoints[k_label[keypoint_info['Label']]] = [int(i+0.5) for i in keypoint_info['Position'][:2]] + [2]

    # 分割信息
    seg_label = {7: 'left', 8: 'right'}
    for poly_info in json_data['Polys'][0]['Shapes']:
        polygon = []
        for point in poly_info['Points']:
            polygon.extend(point['Pos'][:2])
        segmentations[seg_label[poly_info['labelType']]] = polygon

    # bbox信息
    for i in ['left', 'right']:
        points = np.array(segmentations[i] + keypoints[i][:2])
        x = points[::2]
        y = points[1::2]
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        bbox[i] = [x_min, y_min, x_max-x_min, y_max-y_min]
        area[i] = (x_max-x_min) * (y_max-y_min)

    return bbox, keypoints, segmentations, titai, area


if __name__ == '__main__':
    """
    coco的categories是对box的类别
    将整个数据集转为coco类型的思路有两种
    1. 将左右两个肌肉组织作为一个类别，所有关键点只有一种
    --> 适合做topdown，每张图检测两个box， 每个box的heatmap的输出通道为1
    --> 如果也用来做bottom up，由于关键点只有一种，所以每张图只会输出一个heatmap通道，上面有两个响应
    2. 将左右两个肌肉组织作为一整个box，所有存在两种关键点--> 适合做bottom up，且输出的heatmap通道为2
    """

    json_root = '../datas/COCO_style/jsons'
    save_root = '../datas/COCO_style/annotations'
    random.seed(2023)
    val_rate, test_rate = 0, 0.3
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 解析root文件夹下的所有json文件，生成coco格式
    images, annotations = collect_jsons(json_root)

    # shuffle
    all_index_list = [i for i in range(len(images))]
    random.shuffle(all_index_list)

    # save json file
    test_ind, val_ind = int(len(images) * test_rate), int(len(images) * (val_rate+test_rate))
    save_coco_json_dataset([images[i] for i in all_index_list[:test_ind]],
                           [annotations[i] for i in all_index_list[:test_ind]],
                           os.path.join(save_root, 'test.json'))
    if val_rate != 0:
        save_coco_json_dataset([images[i] for i in all_index_list[test_ind+1:val_ind]],
                               [annotations[i] for i in all_index_list[test_ind+1:val_ind]],
                               os.path.join(save_root, 'val.json'))
    save_coco_json_dataset([images[i] for i in all_index_list[val_ind+1:]],
                           [annotations[i] for i in all_index_list[val_ind+1:]],
                           os.path.join(save_root, 'train.json'))
    print('success generate COCO annotations')



