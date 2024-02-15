import glob
import os
import tarfile
import shutil
import json
import SimpleITK as sitk
import cv2
import numpy as np


def tar_extract(tar_path, target_path):
    tar = tarfile.open(tar_path)
    file_names = tar.getnames()
    assert len(file_names) == 2, os.path.basename(tar_path)
    for file_name in file_names:
        tar.extract(file_name, target_path)
    tar.close()
    # os.remove(tar_path)


def get_titai(file_name, titai):
    posture = ['PP', 'SP', 'ST']
    position = ['B3', 'U3', 'U5', 'UE']
    temp = {}
    titai_name = file_name.split('Label')[-1]

    if len(titai_name) == 0:
        return
    posture_ = titai_name[:2]
    position_ = titai_name[2:]
    assert posture_ in posture and position_ in position, file_name
    name = file_name.split('_jpg')[0].split('_png')[0].split('_JPG')[0]
    titai[name] = {'posture': posture_, 'position': position_}


def cope_img(file_path, save_path, file_name, ext, pair_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copyfile(file_path, os.path.join(save_path, file_name) + '.png')
    # save to pair path
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        shutil.copyfile(file_path, os.path.join(pair_path, file_name) + '.png')


def cope_json(json_path, save_path, file_name, titai, pair_path=None, error_info=None, all_titai=None):
    """
    将图片的json文件使用模板初始化为标准格式
    更改内容有：FileInfo:Name、Height、Width， FileName, LabelGroup,
              Models:AngleModel、BoundingBoxLabelModel、FrameLabelModel、MeasureModel、ColorLabelTableModel
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    posture_label = {'PP': 9, 'SP': 10, 'ST': 11}
    position_label = {'B3': 12, 'U3': 13, 'U5': 14, 'UE': 15}

    # load jsons
    with open(json_path, encoding='utf8') as json_:
        json_data = json.load(json_)
    with open('./json_template.json', encoding='utf8') as json_:
        json_template = json.load(json_)

    # 更改文件信息
    json_data['FileInfo']['Name'] = file_name + '.png'
    json_data['FileInfo']['Version'] = "v2.6.0"
    json_template['FileInfo'] = json_data['FileInfo']
    json_template['FileName'] = file_name + '_png'

    # 老版标注，titai存在
    if titai and not ('FrameLabelModel' in json_data['Models'] and json_data['Models']['FrameLabelModel'] is not None):
        if 'class LandMarkListModel * __ptr64' in json_data['Models']:   # 老版本标注文件
            landmark_type = 'class LandMarkListModel * __ptr64'
        elif 'LandMarkListModel' in json_data['Models']:    # 老版标注，但被新版重写了
            landmark_type = 'LandMarkListModel'

        # 获取关键点和poly信息
        landmarks = {}
        for i in json_data['Models'][landmark_type]['Points'][0]['LabelList']:
            landmarks[i['Label']] = i['Position']
        polys = {}
        for i in json_data['Polys'][0]['Shapes']:
            polys[i['labelType']] = i['Points']

        # 更改关键点和poly
        for label, landmark in landmarks.items():
            for i in json_template['Models']['LandMarkListModel']['Points'][0]['LabelList']:
                if i['Label'] == label+3:
                    i['Position'] = landmark
        for label, poly in polys.items():
            for i in json_template['Polys'][0]['Shapes']:
                if i['labelType'] == label+2:
                    i['Points'] = poly
        # 更改体态信息
        json_template['Models']['FrameLabelModel']['ChildFrameLabel'][0]['Label'] = posture_label[titai['posture']]
        json_template['Models']['FrameLabelModel']['ChildFrameLabel'][1]['Label'] = position_label[titai['position']]
    # 老版标注，titai不存在
    elif titai is None and ('class LandMarkListModel * __ptr64' in json_data['Models'] or
                            ('FrameLabelModel' in json_data['Models'] and json_data['Models']['FrameLabelModel'] is None)):
        error_info['old_ann_no_pos'].append(file_name)
        return
        # raise file_name
    # 新版标注
    else:
        json_template['Models'] = json_data['Models']
        json_template['Polys'] = json_data['Polys']

    # 保存体态信息，用于后续检查整个文件夹是否有12个体态
    titai_info = []  # 9-11: posture, 12-15: position
    for frame_label in json_template['Models']['FrameLabelModel']['ChildFrameLabel']:
        titai_info.append(frame_label['Label'])
    all_titai[file_name] = titai_info

    # 将医生标注的左右互换一下，直观一点
    keypoints = {5: None, 6: None}
    polys = {7: [], 8: []}
    for kp in json_template['Models']['LandMarkListModel']['Points'][0]['LabelList']:
        kp['Label'] = 6 if kp['Label'] == 5 else 5
        keypoints[kp['Label']] = kp['Position'] if kp['Position'] is not None else None
    for poly in json_template['Polys'][0]['Shapes']:
        poly['labelType'] = 7 if poly['labelType'] == 8 else 8
        if poly['Points'] is None:
            break
        for temp in poly['Points']:
            polys[poly['labelType']].append(temp['Pos'][:2]) if temp is not None else None

    # 检查标注数据的完整性
    if None in keypoints.values() or len(polys[7]) == 0 or len(polys[8]) == 0:
        error_info['no_full_anno'].append(file_name)
        return
    # 检查标注数据是否正确，左边横坐标小于右边
    for ind in polys.keys():
        polys[ind] = np.array(polys[ind])
    if keypoints[5][0] > keypoints[6][0] or polys[7].T[0].max() > polys[8].T[0].min():
        error_info['ann_error'].append(file_name)
        return

    # 写入json文件
    with open(os.path.join(save_path, file_name + '.json'), 'w') as f:
        json.dump(json_template, f)
    # save to pair path
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        with open(os.path.join(pair_path, file_name + '_png_Label.json'), 'w') as f:
            json.dump(json_template, f)


def cope_mask(nii_path, save_path, file_name, binary_TF=False, pair_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    itk_img = sitk.ReadImage(nii_path)
    if binary_TF:
        itk_img = sitk.Cast(sitk.RescaleIntensity(itk_img), sitk.sitkUInt8)  # 转换成0-255的二值灰度图
    img_array = sitk.GetArrayFromImage(itk_img)
    # 二值化后，img_array的取值为0和255；
    # 未二值化前，img_array的取值为0、45和46（其中45和46分别为图中两种不同标签标签的类别id值）。
    save_name = os.path.join(save_path, file_name)
    save_name = save_name + '_255.png' if binary_TF else save_name + '.png'
    cv2.imwrite(save_name, img_array)
    if pair_path:
        if not os.path.exists(pair_path):
            os.makedirs(pair_path)
        shutil.copy(nii_path, os.path.join(pair_path, file_name) + '_png_Label.nii.gz')


def check_all_titai(all_titai, error_info):
    """
    检查改文件夹是否还有12种体态信息
    :return:
    """
    posture_label = {i: j for j, i in {'PP': 9, 'SP': 10, 'ST': 11}.items()}
    position_label = {i: j for j, i in {'B3': 12, 'U3': 13, 'U5': 14, 'UE': 15}.items()}
    titais = {posture_label[i] + '_' + position_label[j]: [] for i in posture_label for j in position_label}

    # 获取titai信息
    for name, titai in all_titai.items():
        posture, position = '', ''
        for i in titai:
            if i in posture_label:
                posture = posture_label[i]
            if i in position_label:
                position = position_label[i]
        if posture + '_' + position in titais:
            titais[posture + '_' + position].append(name)

    # 检查体态信息
    error_titai = {}
    for titai, data in titais.items():
        if len(data) != 1:
            if len(data) == 0:
                error_titai[titai] = 0
            else:
                error_titai[titai] = [i.split('__')[-1] for i in data]
    if error_titai:
        error_info['no_full_anno'].append({name.split('__')[0]: error_titai})


if __name__ == '__main__':
    root = '../../datas/IRD/data_backup/old-12/COCO_style'
    img_root = '../../datas/IRD/{}/images'.format(root.split('/')[-1])
    json_root = '../../datas/IRD/{}/jsons'.format(root.split('/')[-1])
    mask_root = '../../datas/IRD/{}/masks'.format(root.split('/')[-1])
    pair_path = '../../datas/IRD/{}/pair_files'.format(root.split('/')[-1])
    # 标注错误，如：老标注格式没有体态信息，没有完全标注所有信息，标注错误左边大大于右边，
    error_info = {'old_ann_no_pos': [], 'no_full_anno': [], 'ann_error': []}

    file_name_list = os.listdir(root)
    # 对每个患者，共12个体态进行处理
    for dir_name in file_name_list:
        dir_path = os.path.join(root, dir_name)
        images, titai, all_titai = [], {}, {}
        # 获取该患者的所有图片，并对标注文件解压tar包
        for temp in os.listdir(dir_path):
            ext = temp.split('.')[-1]
            if ext in ['jpg', 'png', 'JPG', 'PNG']:
                if len(glob.glob(os.path.join(dir_path, temp.split('.')[0] + '_*.tar'))) == 1:
                    images.append(temp)
            if ext in ['tar']:
                # 判断医生是否在文件名上注明了体态
                get_titai(temp.split('.')[0], titai)
                tar_extract(os.path.join(dir_path, temp), dir_path)

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
                error_info['old_ann_no_pos'].append(final_name)
                continue
            # 判断标注格式上是否有
            cope_img(os.path.join(dir_path, image_name), img_root, final_name, ext, pair_path=pair_path)
            json_titai = None if base_name not in titai else titai[base_name]
            cope_json(json_file[0], json_root, final_name, json_titai, pair_path=pair_path, error_info=error_info, all_titai=all_titai)
            cope_mask(mask_file[0], mask_root, final_name, binary_TF=True, pair_path=pair_path)
            # cope_mask(mask_file[0], mask_root, final_name, binary_TF=True)
        check_all_titai(all_titai, error_info)   # 检查该文件夹是否有12个体态
    for i, j in error_info.items():
        if i == 'no_full_anno' and len(j) > 0:
            print(i)
            for k in j:
                print(k)
        else:
            print(i, j)
