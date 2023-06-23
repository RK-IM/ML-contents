from pathlib import Path
import xml.etree.ElementTree as et

import pandas as pd
from tqdm.auto import tqdm

data_path = Path('../VOCdevkit/VOC2012')
annot_path = data_path / "Annotations"
annot_list = list(annot_path.glob('./*.xml'))

df = pd.DataFrame()

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

df.to_csv(str(data_path) + '/bbox_dataframe.csv', index=False)