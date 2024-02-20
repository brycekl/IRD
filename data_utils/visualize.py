import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_img(img, target, pre_target=None, task='poly', title='', save_path=None):
    """
    可视化target, target中landmark和mask显示红色，pre_target中landmark和mask显示绿色
    para:
        img: the img for show the result
        target: C*H*W, {'landmark': {...}, 'mask': ...}
    """
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())

    # visualize target

    if task in ['poly', 'all']:
        mask_gt = target['mask'][-2:]
        for mask_gt_item in mask_gt:
            mask_ind = np.where(mask_gt_item == 1)
            img[mask_ind[0], mask_ind[1], 0] = img[mask_ind[0], mask_ind[1], 0] * 0.5 + 0.5  # 花式索引
        if pre_target:
            mask_pre = pre_target['mask'][-2:]
            for mask_gt_item in mask_pre:
                mask_ind = np.where(mask_gt_item == 1)
                img[mask_ind[0], mask_ind[1], 1] = img[mask_ind[0], mask_ind[1], 1] * 0.5 + 0.5

    # should first visualize mask and then landmark
    if task in ['landmark', 'all']:
        landmark_gt = {i: [int(j[0]+0.5), int(j[1]+0.5)] for i, j in target['landmark'].items()}
        for point in landmark_gt.values():
            cv2.circle(img, point, 2, (1, 0, 0), -1)
        if pre_target:
            landmark_pre = {i: [int(j[0] + 0.5), int(j[1] + 0.5)] for i, j in pre_target['landmark'].items()}
            for point in landmark_pre:
                cv2.circle(img, point, 2, (0, 1, 0), -1)

    plt.title(title)
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imsave(os.path.join(save_path, title + '.png'), img)
        plt.close()
    else:
        plt.imshow(img)
        plt.show()
