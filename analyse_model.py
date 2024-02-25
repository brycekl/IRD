import os
import json
import pandas as pd

root = './model'
analyse_attr = {'landmark': ['lr', 'base_c', 'base_size', 'var', 'max_value'],
                'poly': ['lr', 'base_c', 'base_size', 'other_data'],
                'all': []}
result = {train_type: {attr: [] for attr in analyse_attr[train_type] + ['path']} for train_type in analyse_attr}
result['landmark']['mse'] = []
result['poly']['dice'] = []

for train_date in os.listdir(root):
    for train_type in os.listdir(os.path.join(root, train_date)):
        for item in os.listdir(os.path.join(root, train_date, train_type)):
            item_path = os.path.join(root, train_date, train_type, item)
            # get model config data
            with open(os.path.join(root, train_date, train_type, item, 'config.json')) as reader:
                model_config = json.load(reader)
            task = model_config['task']

            # save common para
            for attr in analyse_attr[task]:
                result[task][attr].append(model_config[attr] if model_config.get(attr) else None)
            result[task]['path'].append(item_path)

            # landmark
            if task == 'landmark':
                result[task]['mse'].append(item.split('_')[-1])
            elif task == 'poly':
                result[task]['dice'].append(item.split('_')[-1])
            else:
                pass

# save result
for data_type, res in result.items():
    if res['path']:
        df = pd.DataFrame(res)
        df.to_excel(os.path.join(root, data_type + '.xlsx'), index=False)
