
import numpy as np
import pandas as pd

# 배치 크기만큼 데이터를 넘겨주기 위해 데이터 로더 정의
class DataLoader:
    """Takes the Numpy array Dataset and return iterable data by the batch size.
    Args:
        df (numpy.array): Input Dataset to create iterable data.
            Last column must contain target data.
        batch_size (int): The number of samples to load on each batch.
        shuffle (bool): If 'shuffle=True', data reshuffled at every epoch.
    """
    def __init__(self,
                 df:pd.DataFrame,
                 batch_size:int=4,
                 shuffle:bool=False):
        self.data = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))

    def __len__(self):
        """Get length of DataLoader"""
        return (len(self.data)-1) // self.batch_size + 1

    def getitem(self):
        """Gernerate the iterable data by the batch size.
        Set `shuffle=True` when called the DataLoader makes data shuffles every epoch.
        
        Returns:
            X (list[list]): Features of dataset. shape: [batch_size, features]
            y (list): Target of dataset. shape: [batch_size]
        """
        if self.shuffle: # shuffle이 참이라면 인덱스를 섞어준다. 섞인 인덱스에 해당하는 데이터를 가져온다. 이로인해 에포크마다 배치의 데이터가 섞이게 된다.
            np.random.shuffle(self.indices) # 인덱스 섞기
        for i in range((len(self.data)-1)//self.batch_size + 1): # 데이터 수 / 배치사이즈 만큼 반복한다. 
            indices = self.indices[i*self.batch_size : (i+1)*self.batch_size] # 배치사이즈 만큼의 인덱스를 가져온다.
            X = self.data[indices, :-1] # 섞은 인덱스에 해당하는 값 중 타겟인 마지막 열의 데이터를 제외해서 가져온다.
            y = self.data[indices, -1].reshape(-1, 1) # 마찬가지로 타겟인 마지막 열의 데이터만 가져온다.
            yield X, y # 반복문이 사용 될 때 다음 배치를 가져올 수 있도록 generator로 만들어준다.

def train_test_split(data, train_split=0.8, shuffle=True):
    """Split the data into train and test dataset.
    `train_split` decide the training dataset size.
    Shuffle the dataset before split dataframe into training and test dataset if `shuffle=True`.

    Args:
        data (pandas.DataFrame): Target DataFrame to split.
        train_split (float): Percentage of training data. [0, 1]
        shuffle (bool): Set to True, dataset shuffled before splitting.

    Returns:
        train (np.array): Training Dataset. 
            Length of `train` is train_split times original dataframe length
        test (np.array): Test Dataset.
    """
    shuffled_index = np.random.permutation(len(data)) # 데이터 길이만큼 순서가 섞인 인덱스 배열을 만들어준다.
    split_index = int(train_split * len(data)) # 훈련 데이터 사이즈를 계산한다.

    train_indices = shuffled_index[:split_index] # 훈련 데이터 사이즈를 기준으로 섞은 인덱스 배열에서 훈련용 인덱스를 가져온다.
    test_indices = shuffled_index[split_index:] # 마찬가지로 테스트용 인덱스를 가져온다.

    data = data.to_numpy()
    train = data[train_indices] # 위에서 구한 훈련 인덱스에 해당하는 데이터를 훈련데이터로 지정한다.
    test = data[test_indices] # 테스트 인덱스의 데이터를 테스트 데이터로 지정한다.
    return train, test


class Standardizaion:
    """
    Standardize data along the feature axis.
    First use `fit` method to calculate mean and standard deviation of data.
    And use `transform` to get standardized data
    """
    def __init__(self):
        pass

    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.mean(axis=0, keepdims=True)

    def transform(self, data):
        return (data - self.mean) / self.std
