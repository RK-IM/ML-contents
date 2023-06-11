"""
Train binary classification model.
"""

import get_data, data_setup, model_builder, train, utils


# 하이퍼 파라미터 지정
EPOCHS = 500
BATCH_SIZE = 4

MODEL_DIR = 'model'
MODEL_NAME = 'binary_classifier.pickle'

hidden_dim = 20
out_features = 1

# 데이터 불러오기
binary_df = get_data.download_data()

# 훈련, 테스트 데이터 분리 + 섞기
train_df, test_df = data_setup.train_test_split(binary_df)

# 데이터 표준화
std_scaler = data_setup.Standardizaion()
std_scaler.fit(train_df[:, :-1])
train_df[:, :-1] = std_scaler.transform(train_df[:, :-1])
test_df[:, :-1] = std_scaler.transform(test_df[:, :-1])

# 배치 사이즈만큼 데이터를 넘겨주는 훈련, 테스트 데이터로더 객체 생성
train_dataloader = data_setup.DataLoader(train_df, 
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
test_dataloader = data_setup.DataLoader(test_df,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

# 모델 생성
model = model_builder.LinearModel(in_features=binary_df.shape[1]-1,
                                  hidden_dim=hidden_dim,
                                  out_features=out_features)

# 손실 함수 정의
loss_fn = train.BCELoss()
# optimizer 정의
optimizer = train.SGD(lr=0.001)
# 정확도 계산 함수 정의
accuracy_fn = train.Accuracy

# 훈련 과정 진행
results = train.train(model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     accuracy_fn=accuracy_fn,
                     epochs=EPOCHS)

utils.save_model(model, MODEL_NAME, MODEL_DIR)
