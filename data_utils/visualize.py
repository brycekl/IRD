import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_result(img, target, pre_target=None, task='poly', title='', save_path=None, show=False):
    """
    可视化target, target中landmark和mask显示红色，pre_target中landmark和mask显示绿色
    para:
        img: the img for show the result
        target: C*H*W, {'landmark': {...}, 'mask': ...}
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())
    poly_factor = 0.75 if pre_target else 0.5

    mask_gt = target['mask'][-2:]
    for mask_gt_item in mask_gt:
        mask_ind = np.where(mask_gt_item == 1)
        img[mask_ind[0], mask_ind[1], 0] = img[mask_ind[0], mask_ind[1], 0] * poly_factor + (1 - poly_factor)  # 花式索引
    if task in ['poly', 'all'] and pre_target:
        mask_pre = pre_target['mask'][-2:]
        for mask_pre_item in mask_pre:
            mask_ind = np.where(mask_pre_item == 1)
            img[mask_ind[0], mask_ind[1], 1] = img[mask_ind[0], mask_ind[1], 1] * poly_factor + (1-poly_factor)
        # 如果task为poly，则为poly绘制腹直肌间距
        if task == 'poly' and 'keypoint' in pre_target:
            keypoint = pre_target['keypoint']
            p_dis = pre_target['p_dis']
            cv2.line(img, keypoint['gt'][5], keypoint['gt'][6], (1, 0, 0), 2)
            cv2.line(img, keypoint['pre'][5], keypoint['pre'][6], (0, 1, 0), 2)
            cv2.putText(img, f'l_mse: {round(pre_target["p_mse"][5], 2)}mm', [20, img.shape[0] - 60],
                        cv2.FONT_HERSHEY_COMPLEX, 1, (1, 0, 1), 2)
            cv2.putText(img, f'r_mse: {round(pre_target["p_mse"][6], 2)}mm', [20, img.shape[0] - 30],
                        cv2.FONT_HERSHEY_COMPLEX, 1, (1, 0, 1), 2)
            cv2.putText(img, f'GT: {round(p_dis["p_dis_gt"], 2)}mm', [350, img.shape[0] - 60],
                        cv2.FONT_HERSHEY_COMPLEX, 1, (1, 0, 0), 2)
            cv2.putText(img, f'Pre: {round(p_dis["p_dis_pre"], 2)}mm', [350, img.shape[0] - 30],
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 1, 0), 2)

    # should first visualize mask and then landmark
    if task in ['landmark', 'all']:
        landmark_gt = {i: [int(j[0]+0.5), int(j[1]+0.5)] for i, j in target['landmark'].items()}
        for point in landmark_gt.values():
            cv2.circle(img, point, 2, (1, 0, 0), -1)
        if pre_target:
            landmark_pre = {i: [int(j[0] + 0.5), int(j[1] + 0.5)] for i, j in pre_target['landmark'].items()}
            for point in landmark_pre.values():
                cv2.circle(img, point, 2, (0, 1, 0), -1)
            if 'l_mse' in pre_target:
                keypoint = {'gt': target['landmark'], 'pre': pre_target['landmark']}
                l_dis = pre_target['l_dis']
                cv2.line(img, keypoint['gt'][5], keypoint['gt'][6], (1, 0, 0), 2)
                cv2.line(img, keypoint['pre'][5], keypoint['pre'][6], (0, 1, 0), 2)
                cv2.putText(img, f'l_mse: {round(pre_target["l_mse"][5], 2)}mm', [20, img.shape[0] - 60],
                            cv2.FONT_HERSHEY_COMPLEX, 1, (1, 0, 1), 2)
                cv2.putText(img, f'r_mse: {round(pre_target["l_mse"][6], 2)}mm', [20, img.shape[0] - 30],
                            cv2.FONT_HERSHEY_COMPLEX, 1, (1, 0, 1), 2)
                cv2.putText(img, f'GT: {round(l_dis["l_dis_gt"], 2)}mm', [350, img.shape[0] - 60],
                            cv2.FONT_HERSHEY_COMPLEX, 1, (1, 0, 0), 2)
                cv2.putText(img, f'Pre: {round(l_dis["l_dis_pre"], 2)}mm', [350, img.shape[0] - 30],
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 1, 0), 2)

    plt.title(title)
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imsave(os.path.join(save_path, title + '.png'), img)
        plt.close()
    if show:
        plt.imshow(img)
        plt.show()
