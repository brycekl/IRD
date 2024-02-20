import os
import json
import pandas as pd

root = './model'
analyse_attr = {'landmark': ['lr', 'base_c', 'base_size', 'var', 'max_value'], 'poly': [], 'all': []}
result = {train_type: {attr: [] for attr in analyse_attr[train_type] + ['path']} for train_type in analyse_attr}
result['landmark']['mse'] = []
result['landmark']['path'] = []

for train_date in os.listdir(root):
    if train_date != '20240218':
        continue
    for train_type in os.listdir(os.path.join(root, train_date)):
        for item in os.listdir(os.path.join(root, train_date, train_type)):
            item_path = os.path.join(root, train_date, train_type, item)
            # get model config data
            with open(os.path.join(root, train_date, train_type, item, 'config.json')) as reader:
                model_config = json.load(reader)

            # landmark
            if train_type == 'landmark':
                for attr in analyse_attr[train_type]:
                    result[train_type][attr].append(model_config[attr] if model_config.get(attr) else None)
                result[train_type]['path'].append(item_path)
                result[train_type]['mse'].append(item.split('_')[-1])
            elif train_type == 'poly':
                pass
            else:
                pass

# save result
for data_type, res in result.items():
    if data_type == 'landmark':
        df = pd.DataFrame(res)
        df.to_excel(os.path.join(root, data_type + '.xlsx'), index=False)
