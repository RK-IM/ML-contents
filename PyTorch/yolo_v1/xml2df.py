from pathlib import Path
import xml.etree.ElementTree as et

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from params import DATA_PATH, ANNOT_PATH


df = pd.DataFrame()

annot_list = list(ANNOT_PATH.glob('./*.xml'))
tk0 = tqdm(annot_list, total=len(annot_list))
for annot in tk0:
    xtree = et.parse(annot)
    xroot = xtree.getroot()
    xml_dict = {
        'id': [],
        'class': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        }

    for node in xroot:
        if node.tag == 'filename':
            name = node.text.split('.')[0]
        
        if node.tag == 'size':
            temp = {}
            for elem in node:
                temp[elem.tag] = elem.text
            break

    for node in xroot:
        if node.tag == 'object':
            
            obj_class = node.find('name').text

            xml_dict['id'].append(name)
            xml_dict['class'].append(obj_class)
            for k, v in temp.items():
                xml_dict[k] = v

            for elem in list(node.find('bndbox')):
                xml_dict[elem.tag].append(elem.text)

    df = pd.concat(
        [df,
         pd.DataFrame.from_dict(xml_dict)],
         axis=0)


print("[INFO] Split dataset into 10 folds...")
df = df.reset_index(drop=True)
df['fold'] = -1

sgkf = StratifiedGroupKFold(n_splits=10)
for i, (tr_idx, vl_idx) in enumerate(sgkf.split(df, 
                                                y=df['class'], 
                                                groups=df['id'])):
    df.loc[vl_idx, 'fold'] = i
print("[INFO] Done!")

print("[INFO] Split each fold into training and validation set...")
df_with_train = pd.DataFrame()
for i in range(10):
    temp_df = df.query("fold==@i")
    temp_df = temp_df.reset_index(drop=True)
    temp_df['train'] = False

    sgkf = StratifiedGroupKFold(n_splits=5)
    for tr_idx, vl_idx in sgkf.split(X=temp_df,
                                     y=temp_df['class'],
                                     groups=temp_df['id'],):
        temp_df.loc[tr_idx, 'train'] = True
        break

    df_with_train = pd.concat(
        [df_with_train, temp_df],
        axis=0
    )

df_with_train = df_with_train.reset_index(drop=True)
print("[INFO] Done!")
df_with_train.to_csv(str(DATA_PATH) + '/bbox_dataframe.csv', index=False)
print(f"[INFO] Label info csv saved at {DATA_PATH}/bbox_dataframe.csv")
