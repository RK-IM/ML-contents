from pathlib import Path
import xml.etree.ElementTree as et

import pandas as pd
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

df.to_csv(str(DATA_PATH) + '/bbox_dataframe.csv', index=False)
