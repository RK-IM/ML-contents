import os
import re
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample

from skmultilearn.model_selection import iterative_train_test_split

from typing import List, Tuple

def preprocessing(string: str) -> str:
    string = re.sub('\n', ' ', string)
    string = re.sub(r'[ㄱ-ㅎㅏ-ㅣ ]', ' ', string)
    string = re.sub(r'[^0-9가-힣,.!? ]', '', string)
    string = re.sub(r' *[.!?]+ *', '.', string)
    string = re.sub('\.([0-9]+)\.', '', string)
    string = re.sub(r'^[. ]+', '', string)
    string = re.sub(' +', ' ', string)
    return string

def get_data(data_path: str, file_name: str, column='review') -> pd.DataFrame:
    """
    Args:
        data_path (str): directory which csv file lives.
        file_name (str): file name with extension to load.

    Returns:
        df (pandas.DataFrame): columns with 'review'(str),
            and target (15 one-hot) 
    """
    df = pd.read_csv(os.path.join(data_path, file_name))
    # df = pd.concat([df['review'], df.loc[:, df.dtypes==int]], axis=1)
    df[column] = df[column].map(preprocessing)
    return df

def data_split(dataframe: pd.DataFrame,
               test_size: float = 0.3,
               random_state: int = None) -> Tuple[np.array]:
    """
    Use `iterative_train_test_split` in scikit-multilearn library
    to split dataframe into training set and test set
    with feature(text) and target(15 one-hot)
    because of extremely imbalanced data.

    Args: 
        dataframe (pandas.DataFrame): dataframe trying to split.
        random_state (int)
        
    Returns:
        X_train (numpy.array): 
            Train text data array with index (index, text).
        y_train (numpy.array): Train target data (15 one-hot).
        X_test (numpy.array):
            Test text data array with index (index, text).
        y_test (numpy.array): Test target data (15 one-hot).
    """
    if random_state is not None:
        np.random.seed(random_state)
    X_array = np.array(dataframe[['review']].reset_index())
    y_array = np.array(dataframe.loc[:, dataframe.dtypes==int])

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X_array, y_array, test_size=test_size
    )
    return X_train, y_train, X_test, y_test

def long_form(dataframe: pd.DataFrame,
              train_index: List) -> pd.DataFrame:
    """
    Extract train data from train_index.
    And convert train dataframe to long form
    for train SBERT, which takes two sentence as input.

    Args:
        dataframe (pandas.DataFrame): 
            Dataframe for extract training set
        train_index (List): Training set index

    Returns:
        dataframe_melt (pandas.DataFrame):
            Long format of training dataset.
            columns = 'index' (same index is same review)
                      'review' (sentence 1), 
                      'variable' (sentence 2),
                      'value' (target, 0 or 1)
    """
    dataframe = pd.concat([dataframe['review'], dataframe.loc[:, dataframe.dtypes==int]], axis=1)
    dataframe_melt = dataframe.loc[train_index, :]
    dataframe_melt = pd.melt(dataframe_melt.reset_index(),
                             id_vars=['index', 'review'])
    return dataframe_melt

def dataloader_for_sbert(data_path: str,
                         file_name: str,
                         batch_size: str,
                         test_size: float,
                         is_train: bool = True):
    # prepare data
    df = get_data(data_path=data_path, file_name=file_name)
    X_train, _, _, _ = data_split(df, test_size=test_size)
    df_melt = long_form(df, X_train[:, 0])

    # array for inputs
    gold_samples = []

    # Create input for sbert
    for i in range(len(df_melt)):
        gold_samples.append(InputExample(texts=[df_melt.loc[i, 'review'],
                                                 df_melt.loc[i, 'variable']],
                                          label=float(df_melt.loc[i, 'value'])))

    # Set DataLoader
    dataloader = DataLoader(gold_samples,
                            shuffle=True,
                            batch_size=batch_size)

    if not is_train:
        return gold_samples
    else:
        return dataloader

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, arrays, labels):
        super().__init__()
        self.arrays = arrays
        self.labels = labels

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx: int):
        x = torch.FloatTensor(self.arrays[idx])
        y = torch.FloatTensor(self.labels[idx])
        return x, y