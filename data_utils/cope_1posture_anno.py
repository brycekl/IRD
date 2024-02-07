import os
import glob
import json
import SimpleITK as sitk
import shutil
import cv2
import numpy as np

from cope_raw_anno import tar_extract, cope_img
from scipy import ndimage


def cope_json(json_path, save_path, file_name, titai=None, pair_path=None, error_info=None, all_titai=None):
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

    # 检查是否有标注两个关键点
    keypoints = {5: None, 6: None}
    for kp in json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']:
        keypoints[kp['Label']] = kp['Position'] if kp['Position'] is not None else None

    if not all(keypoints.values()):
        error_info['no_two_landmark'].append(file_name)

    # 写入json文件
    with open(os.path.join(save_path, file_name + '.json'), 'w') as f:
        json.dump(json_data, f)


def cope_mask(nii_path, save_path, file_name, binary_TF=False, pair_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    itk_img = sitk.ReadImage(nii_path)
    if binary_TF:
        itk_img = sitk.Cast(sitk.RescaleIntensity(itk_img), sitk.sitkUInt8)  # 转换成0-255的二值灰度图
    img_array = sitk.GetArrayFromImage(itk_img)
    # 检查是否标注了两个区域
    label = ndimage.label(img_array)
    num_region = len(np.unique(label[0])) - 1
    if num_region < 2 or num_region != label[1]:
        if num_region == 1 and len(np.unique(img_array)) != 3:
            error_info['no_two_region'].append(file_name)

    # save file
    save_name = os.path.join(save_path, file_name)
    save_name = save_name + '_255.jpg' if binary_TF else save_name + '.jpg'
    cv2.imwrite(save_name, img_array)
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        shutil.copy(nii_path, os.path.join(pair_path, file_name) + '_jpg_Label.nii.gz')


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
    root = '../../datas/IRD/data_backup/gaoya'
    img_root = '../../datas/IRD/{}/images'.format(root.split('/')[-1])
    json_root = '../../datas/IRD/{}/jsons'.format(root.split('/')[-1])
    mask_root = '../../datas/IRD/{}/masks'.format(root.split('/')[-1])
    # 错误类型：标注文件数量错误，单张图片未标注两个点，单张图片未标注两个区域，整个文件夹未标注四个position
    error_info = {'anno_num_error': [], 'no_two_landmark': [], 'no_two_region': [], 'no_all_position': {}}

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
            cope_img(os.path.join(dir_path, image_name), img_root, final_name, ext)
            cope_mask(mask_file[0], mask_root, final_name, binary_TF=True)
            cope_json(json_file[0], json_root, final_name, error_info=error_info, all_titai=all_titai)
        no_exist = check_titai(all_titai)
        if no_exist:
            error_info['no_all_position'][dir_name] = no_exist

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



