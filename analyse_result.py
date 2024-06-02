import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def calculate_confusion_matrix(GT, PRE, ):
    smooth = 1e-9
    tp = np.sum((GT == 1) & (PRE == 1))
    fp = np.sum((GT == 0) & (PRE == 1))
    tn = np.sum((GT == 0) & (PRE == 0))
    fn = np.sum((GT == 1) & (PRE == 0))
    cm = confusion_matrix(GT, PRE, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['separated', 'unseparated'])
    disp.plot()
    plt.show()
    return tp+fn, fp+tn, (tp+tn+smooth)/(tp+tn+fp+fn+smooth), (tp+smooth)/(tp+fp+smooth), (tp+smooth)/(tp+fn+smooth)


def main(root):
    data = pd.read_excel(f'{root}/result.xlsx').to_dict('list')
    names_index = sorted(zip(data['name'], range(len(data['name']))), key=lambda item: item[0])
    res = {}

    # 距离类型，直接预测landmark的l_dis，预测poly后取的p_dis
    for cope_name in ['l_dis', 'p_dis']:
        if f'{cope_name}_gt' in data:
            gt_dis = np.array(data[f'{cope_name}_gt'])
            pre_dis = np.array(data[f'{cope_name}_pre'])
            GT, PRE = [], []
            # 一个病例有四个位点的数据
            for i in range(len(names_index)//4):
                names_index_i = names_index[i*4: (i+1)*4]
                base_name = set([item[0].split('__')[0] for item in names_index_i])
                assert len(base_name) == 1
                indexs = [item[1] for item in names_index_i]
                gt_i = gt_dis[indexs]
                pre_i = pre_dis[indexs]
                if any(gt_i > 20):
                    GT.append(1)
                else:
                    GT.append(0)
                if any(pre_i > 20):
                    PRE.append(1)
                else:
                    PRE.append(0)
            res[cope_name] = {'GT': np.array(GT), 'PRE': np.array(PRE)}
    for cope_name, bool_res in res.items():
        pos, neg, acc, precision, recall = calculate_confusion_matrix(bool_res['GT'], bool_res['PRE'])
        print(os.path.basename(path), cope_name)
        print(f'分离：{pos}   未分离：{neg}   acc：{acc}   precision：{precision}    recall：{recall}')
    pass


if __name__ == '__main__':
    path = './result/240530/poly/all_same_od_4-all_same_0.921'
    main(path)
