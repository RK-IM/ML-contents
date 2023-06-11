import os
import requests
from pathlib import Path

import pandas as pd

def download_data():
    """Download 'binary_dataset.csv' if this file not in data directory.
    If already have one, then skip downloading.
    This csv file contains 8 float type features and one binary target at last column.

    Return
        df (pandas.DataFrame): DataFrame with 9 columns. 
            First 8 columns (x1 ~ x8) are feature columns with float type.
            Last 1 column (y) is target column with binary type (0 or 1)
    """
    # 데이터 저장용 data 디렉토리 생용
    data_path = Path('data/')
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)

    try: # 데이터셋이 현재 위치에 있다면 판다스로 불러오기
        df = pd.read_csv(data_path / 'binary_dataset.csv') 
        # print("[INFO] 'binary_dataset.csv' already in directory. Get DataFrame from 'data/'")
    except: # 데이터셋이 현재 위치에 없다면
        with open(data_path / "binary_dataset.csv", "wb") as f: # requests로 url에서 데이터를 가져온 후에 판다스로 불러온다.
            request = requests.get("https://drive.google.com/uc?export=download&id=1SCO0ZGL_EDGWc9Le0JFDw9eww86xk1xJ")
            # print("[INFO] No 'binary_dataset.csv' found in current directory. Downloading dataset...")
            f.write(request.content)
            # print("[INFO] Download Done!")
        # !wget "https://drive.google.com/uc?export=download&id=1SCO0ZGL_EDGWc9Le0JFDw9eww86xk1xJ" -O "binary_dataset.csv"
        # !mv binary_dataset.csv {data_path}
        df = pd.read_csv(data_path / 'binary_dataset.csv')
    return df
