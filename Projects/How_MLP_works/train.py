
import numpy as np
from data_setup import DataLoader
from model_builder import Linear

class BCELoss:
    """
    Calculate binary cross entropy loss.
    Set boundary of `pred` [1e-7, 1-1e-7] to prevent values get close to 0 in log.
    """

    def forward(self, true, pred):
        """
        Args:
            true: True label of dataset.
            pred: The output of the model. Contains the probability of prediction which is positive.
        """
        self.true = true
        self.pred = np.clip(pred, 1e-7, 1-1e-7)

        self.loss = -(self.true * np.log(self.pred) + (1-self.true) * np.log(1-self.pred))
        return self.loss

    def backward(self, dvalue):
        self.dvalue = np.clip(dvalue, 1e-7, 1-1e-7)
        return -(self.true / self.dvalue - (1 - self.true) / (1 - self.dvalue))


class SGD:
    """Stochastic Gradient Descent. Use learning rate and derivatives of layers,
    update parameters to minimize the loss. Learning rate determines how much of derivatives 
    effect the parameter changes.
    """
    def __init__(self,
                 lr=1e-3):
        self.lr = lr

    def zero_grad(self, model):
        for layer in model.layers[::-1]:
            if isinstance(layer, Linear):
                layer.dweights = 0
                layer.dbiases = 0

    def step(self, model):
        for layer in model.layers[::-1]:
            if isinstance(layer, Linear):
                layer.weights -= (self.lr * layer.dweights)
                layer.biases -= (self.lr * layer.dbiases)


def Accuracy(true: np.array,
             pred: np.array):
    """Calcuate Accuracy.
    Get true label and prediction probability, and round probability to compare with true which data type is integer.
    Args:
        true: True label of dataset.
        pred: The output of the model. Contains the probability of prediction which is positive.

    Returns: Accuracy
    """
    acc = sum(pred.round()==true)
    return acc[0]


def train_step(model,
               dataloader:DataLoader,
               loss_fn,
               optimizer,
               accuracy_fn):
    """
    Get data from DataLoader, run through training step which contains 
    (forward pass, loss calculation, optimzier initialize, back propagation, optimizer step).
    
    Args:
        model: Linear model to train.
        dataloader: Data generator, to be trained on model.
        loss_fn: Loss function to minimize.
        optimizer: Update parameters by using loss_fn and layers backward methods.
        accuracy_fn: Function to calculate accuracy.

    Returns:
        train_loss: Overall loss on one epoch using training data.
        train_acc: Overall accuracy on one epoch using training data.
    """
    train_loss, train_acc = 0, 0 # 훈련 결과 초기화

     # 훈련 데이터로더로 훈련 데이터 가져오기
    for X, y in dataloader.getitem():

        # 순전파를 통한 확률 예측
        y_pred_prob = model.forward(X)

        # 손실 계산
        loss = loss_fn.forward(y, y_pred_prob)

        # optimizer 초기화
        optimizer.zero_grad(model)

        # 손실함수 역전파
        dvalues = loss_fn.backward(y_pred_prob)
        # 모델 역전파
        model.backward(dvalues)

        # 모델 학습 가능한 파라미터 업데이트
        optimizer.step(model)

        # 배치사이즈 만큼 더해졌으므로 그만큼 나눠준다. 훈련 손실과 정확도 업데이트
        train_loss += loss.sum() / dataloader.batch_size
        train_acc += accuracy_fn(y, y_pred_prob) / dataloader.batch_size

    # 데이터로더의 길이만큼 더해졌으므로 그만큼 손실과 정확도를 나눠준다.
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model,
              dataloader,
              loss_fn,
              accuracy_fn):
    """
    Get data from DataLoader, run through test step which contains (forward pass, loss calculation). 
    Unlike training step, test step doesn't have optimizer step and back propagation.

    Args:
        model: Linear model to be tested.
        dataloader: Data generator, to be tested on model.
        loss_fn: Loss function to minimize.
        accuracy_fn: Function to calculate accuracy.

    Returns:
        train_loss: Overall loss on one epoch using test data.
        train_acc: Overall accuracy on one epoch using test data.
    """
    test_loss, test_acc = 0, 0 # 테스트 결과 초기화

    # 테스트 데이터로더로 테스트 데이터 가져오기
    for X_test, y_test in dataloader.getitem():

        # 순전파를 통한 예측 진행
        test_pred_prob = model.forward(X_test)

        # 손실 계산 
        loss = loss_fn.forward(y_test, test_pred_prob)

        # 훈련 과정과 똑같이 테스트 손실과 정확도 업데이트
        test_loss += loss.sum() / dataloader.batch_size
        test_acc += accuracy_fn(y_test, test_pred_prob) / dataloader.batch_size
            
    # 훈련 과정과 똑같이 테스트 손실과 정확도 업데이트
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          accuracy_fn,
          epochs):
    """
    Pass model through train_step() and test_step(), repeat as many as epochs.
    
    Args:
        model: Linear model to train and to be tested.
        train_dataloader: Dataloader used in train step.
        test_dataloader: Dataloader usesd in test step.
        loss_fn: Loss function to minimize.
        optimizer: Update trainable parameters.
        accuracy_fn: Function to calculate accuracy.
        epochs: How many times training loop repeat.

    Returns:
        results (Dict[str, List]): Results of training step and test step. 
            Loss value and Accuracy for each step for epochs.
    """
    # 결과 초기화
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}
   
    # 에포크마다 반복
    for epoch in range(epochs):
        # 훈련 과정을 진행하고 손실과 정확도 계산
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn)
        # 테스트 과정을 진행하고 손실과 정확도 계산
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn)
        
        # 결과 딕셔너리에 훈련과 테스트 결과(손실, 정확도) 저장
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        # 결과 출력
        if (epoch % (epochs//10) == 0) or (epoch == epochs-1):
            print(f"[Epochs {epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_acc:.4f}")
        
    return results
