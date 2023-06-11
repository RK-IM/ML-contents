from config import CFG
from model_builder import load_sbert, SimpleMLC

import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

import torch

def predict_sample():    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument("--text",
                        help="write a text to predict vibe",
                        type=str)

    args = parser.parse_args()

    tagged_store = pd.read_csv(f"{CFG.DATA_PATH}/{CFG.TAG_CSV}")

    s_bert = load_sbert(model_path=CFG.SBERT_MODEL_PATH,
                        sbert_path=CFG.SBERT_MODEL_FOLDER)

    classifier = SimpleMLC(n_classes=CFG.N_CLASSES)
    classifier.load_state_dict(torch.load(f'{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.pth',
                                          map_location=torch.device(device)))

    vectorized = s_bert.encode(args.text).reshape((1, -1))
    classifier.eval()
    with torch.inference_mode():
        result = classifier(torch.tensor(vectorized))
        result = torch.sigmoid(result)
        result = result.cpu().detach().numpy()

    res = pd.DataFrame(result, columns=CFG.CLASS_NAMES).T
    res = res.rename(columns={0: 'Probability'})

    convert_col = dict(zip(tagged_store.iloc[:, -16:-1].columns, CFG.CLASS_NAMES))
    tagged_store = tagged_store.rename(columns=convert_col)

    score = ((tagged_store.iloc[:, -16:-1] * res['Probability']).div(
        np.sqrt(np.square(tagged_store.iloc[:, -16:-1]).sum(axis=1)) * 
        np.sqrt(np.square(res['Probability']).sum()), axis=0)
    ).sum(axis=1)

    store_tag_prob = pd.concat([tagged_store['name'], tagged_store.iloc[:, -16:-1] * res['Probability']], axis=1)
    store_tag_prob['sum'] = store_tag_prob.iloc[:, -16:-1].sum(axis=1, numeric_only=True)
    store_tag_prob['similarity'] = score


    tags = res[res['Probability']>0.5].index
    if len(tags) > 0:
        print(f"{', '.join(res[res['Probability']>0.5].index)} 태그를 가진 식당들을 보여줍니다")
    else:
        print("좀 더 자세한 분위기가 나타나게 작성해주세요!")


    if store_tag_prob['sum'].max() < 1:
        print("좀 더 자세한 분위기가 나타나게 작성해주세요!")
    else:
        store_tag_prob = store_tag_prob.sort_values('similarity', ascending=False)
        store_tag_prob = store_tag_prob.T.iloc[:, :5]

        store_tag_prob.columns = store_tag_prob.iloc[0]
        store_tag_prob = store_tag_prob.drop(store_tag_prob.index[0])
        print(tabulate(store_tag_prob, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == "__main__":
    predict_sample()
