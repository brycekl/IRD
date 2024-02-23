import os
import glob
import json
import tarfile
import SimpleITK as sitk
import shutil
import numpy as np

from scipy import ndimage
import cv2


def tar_extract(tar_path, target_path):
    tar = tarfile.open(tar_path)
    file_names = tar.getnames()
    assert len(file_names) == 2, os.path.basename(tar_path)
    for file_name in file_names:
        tar.extract(file_name, target_path)
    tar.close()
    # os.remove(tar_path)


def cope_img(file_path, save_path, file_name, ext, pair_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copyfile(file_path, os.path.join(save_path, file_name) + '.png')
    # save to pair path
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        shutil.copyfile(file_path, os.path.join(pair_path, file_name) + '.png')


def cope_json(json_path, save_path, file_name, titai=None, pair_path=None, all_titai=None):
    """
    将图片的json文件使用模板初始化为标准格式
    更改内容有：FileInfo:Name、Height、Width， FileName, LabelGroup,
              Models:AngleModel、BoundingBoxLabelModel、FrameLabelModel、MeasureModel、ColorLabelTableModel
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    posture_label = {'PP': 9}
    position_label = {'B3': 12, 'U3': 13, 'U5': 14, 'UE': 15}

    # load jsons
    with open(json_path, encoding='utf8') as reader:
        json_data = json.load(reader)

    # 保存体态信息，用于后续检查整个文件夹是否有4个体态
    titai_info = []  # 9-11: posture, 12-15: position
    for frame_label in json_data['Models']['FrameLabelModel']['ChildFrameLabel']:
        titai_info.append(frame_label['Label'])
    all_titai[file_name] = titai_info

    # 检查是否有标注两个关键点, 互换5，6的标注
    keypoints = {5: None, 6: None}
    for kp in json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']:
        kp['Label'] = 6 if kp['Label'] == 5 else 5
        keypoints[kp['Label']] = kp['Position'] if kp['Position'] is not None else None

    if not all(keypoints.values()):
        error_info['no_two_landmark'].append(file_name)
    if keypoints[5][0] > keypoints[6][0]:
        error_info['landmark_anno_error'].append(file_name)

    # 写入json文件
    with open(os.path.join(save_path, file_name + '.json'), 'w') as f:
        json.dump(json_data, f)
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        with open(os.path.join(pair_path, file_name + '_png_Label.json'), 'w') as f:
            json.dump(json_data, f)


def cope_mask(nii_path, save_path, file_name, binary_TF=False, pair_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    itk_img = sitk.ReadImage(nii_path)
    if binary_TF:
        itk_img = sitk.Cast(sitk.RescaleIntensity(itk_img), sitk.sitkUInt8)  # 转换成0-255的二值灰度图
    img_array = sitk.GetArrayFromImage(itk_img)
    mask = check_mask(img_array, file_name)

    # save file
    save_name = os.path.join(save_path, file_name + '.png')
    cv2.imwrite(save_name, mask)
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        shutil.copy(nii_path, os.path.join(pair_path, file_name) + '_png_Label.nii.gz')


def check_mask(img_array, file_name=None):
    mask = np.zeros_like(img_array)

    # 检查标注的分割数据
    # 1. 去除误标注的小区域，标注区域连通像素数<10
    # 2. 检查区域个数，若为2，保存
    # 3. 若不为2， 检查标注的像素值，若像素值只有两种，每种像素只取最大区域进行保存
    # 4. 以上都不满足，则报错

    # 去除误标注的小区域后
    label = ndimage.label(img_array)
    label_nums = np.bincount(label[0].flat)
    legal_region_ind = np.where(label_nums > 10)
    legal_region = np.isin(label[0], legal_region_ind).astype(int)
    img_array = img_array * legal_region

    # check mask
    label = ndimage.label(img_array)
    if label[1] == 2:  # two region
        left = 1 if np.median(np.where(label[0] == 1)[1]) < np.median(np.where(label[0] == 2)[1]) else 2
        right = 2 if left == 1 else 1
    # one or more region of poly
    else:
        label_ = np.zeros_like(label[0])
        mask_value = np.unique(img_array)
        if len(mask_value) == 3:
            # just save the biggest region
            for value in mask_value[1:]:
                value_mask = img_array == value
                value_label = ndimage.label(value_mask)
                if value_label[1] > 1:
                    biggest_region = value_label[0] == np.argmax(np.bincount(value_label[0].flat)[1:]) + 1
                    label_[biggest_region] = label_.max() + 1
                else:
                    label_[value_label[0].astype(np.bool_)] = label_.max() + 1
            # save result in poly_mask
            left = 1 if np.median(np.where(label_ == 1)[1]) < np.median(np.where(label_ == 2)[1]) else 2
            right = 2 if left == 1 else 1
        else:
            # if len(legal_region_ind[0]) - 1 != 2 and len(np.unique(img_array)) - 1 != 2:  # 可能会有标注重合的情况
            error_info['no_two_region'].append(file_name)

    mask[label[0] == left] = 1
    mask[label[0] == right] = 2
    return mask


def check_titai(all_titai):
    """
    检查改文件夹是否还有12种体态信息
    :return:
    """
    exist = {12: 'B3', 13: 'U3', 14: 'U5', 15: 'UE'}
    titais = list(all_titai.values())
    for temp in titais:
        if len(temp) == 2:
            posture, position = temp
            if posture == 9 and position in exist:
                exist.pop(position)
    if exist:
        return list(exist.values())
    return False


if __name__ == '__main__':
    """
    used for cope the data with four position of PP posture
    """
    root = '../../datas/IRD/data_backup/267-290gaoya'
    img_root = '../../datas/IRD/{}/images'.format(root.split('/')[-1])
    json_root = '../../datas/IRD/{}/jsons'.format(root.split('/')[-1])
    mask_root = '../../datas/IRD/{}/masks'.format(root.split('/')[-1])
    pair_root = '../../datas/IRD/{}/pair_files'.format(root.split('/')[-1])
    # 错误类型：标注文件数量错误，单张图片未标注两个点，单张图片未标注两个区域，整个文件夹未标注四个position
    error_info = {'anno_num_error': [], 'no_two_landmark': [], 'no_two_region': [], 'no_all_position': {},
                  'landmark_anno_error': []}   # todo 在此处定义的变量为模块级别的变量，在整个模块内可见

    file_name_list = os.listdir(root)
    # 对每个患者，共12个体态进行处理
    for dir_name in file_name_list:
        dir_path = os.path.join(root, dir_name)
        images, all_titai = [], {}
        # 获取该患者的所有图片，并对标注文件解压tar包
        for temp in os.listdir(dir_path):
            ext = temp.split('.')[-1]
            if ext in ['jpg', 'png', 'JPG', 'PNG']:
                if len(glob.glob(os.path.join(dir_path, temp.split('.')[0] + '_*.tar'))) == 1:
                    images.append(temp)
            if ext in ['tar']:
                # 判断医生是否在文件名上注明了体态
                tar_extract(os.path.join(dir_path, temp), dir_path)

        # 开始处理图片
        print('开始处理{}  共{}张图片'.format(dir_name, len(images)))
        # 根据图片名，获取相应的图片，json和mask，进行处理
        for image_name in images:
            base_name = image_name.split('.')[0]
            ext = image_name.split('.')[-1]
            final_name = dir_name + '__' + base_name    # 文件最后重命名的格式

            json_file = glob.glob(dir_path + "/" + base_name + '_*.json')
            mask_file = glob.glob(dir_path + '/' + base_name + '_*.nii.gz')
            # 没有标注,或标注不止一个
            if len(json_file) != 1 and len(mask_file) != 1:
                error_info['anno_num_error'].append(final_name)
                continue
            cope_img(os.path.join(dir_path, image_name), img_root, final_name, ext, pair_path=pair_root)
            cope_mask(mask_file[0], mask_root, final_name, binary_TF=True, pair_path=pair_root)
            cope_json(json_file[0], json_root, final_name, all_titai=all_titai, pair_path=pair_root)
        no_exist = check_titai(all_titai)
        if no_exist:
            error_info['no_all_position'][dir_name] = no_exist

    # 去除已检查的landmark_anno_error,
    with open('./landmark_anno_error', 'r') as reader:
        legal_landmark_error = [line.strip() for line in reader.readlines()]
    error_info['landmark_anno_error'] = list(filter(lambda x: x not in legal_landmark_error, error_info['landmark_anno_error']))

    for i, j in error_info['no_all_position'].items():
        print(i + ': ' + ','.join(j))
    for item in error_info['no_two_landmark']:
        print(*item.split('__'))

    for i, j in error_info.items():
        if i == 'no_full_anno' and len(j) > 0:
            print(i)
            for k in j:
                print(k)
        else:
            print(i, j)



