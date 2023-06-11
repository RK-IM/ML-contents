## 개요

흉부 Xray 사진에서 비정상적인 부분을 찾아내는 Object detecting model 을 만든 프로젝트입니다.

## 데이터

데이터는 캐글에서 진행한 [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)에서 가져왔습니다. dicom형식으로 저장되어있는 약 15000개의 훈련 데이터, 3000개의 테스트 데이터가 있습니다. 

## 파일 설명

- chest-x-rays-part1-bbox-tfrecords.ipynb: dicom 파일에서 이미지 가져오기, 전처리, boundary box 처리, 타겟 비율에 맞게 데이터 나누기, tensorflow object detection api 사용을 위해 tfrecord로 저장하기를 진행했습니다. 구체적인 진행 방법은 노트북 파일 안에 서술되어있습니다.
- chest_x_rays_part2_model.ipynb: 텐서플로의 객체탐지 api를 사용해 학습과 예측을 진행했습니다. 모델은 사용할 수 있는 리소스의 제한으로, 제시되었을 당시 COCO 데이터셋에서 AP 점수가 55.1로 낮은 파라미터로 높은 점수를 냈던, EfficientDet를 사용했습니다. 앞에서 저장한 tfrecord 파일을 가져오고 모델을 수행하는 태스크에 맞게 config 파일도 수정합니다. 예측 결과에 boundary box도 병합 방법들을 비교했습니다. 구체적인 진행 설명은 마찬가지로 노트북 파일에 작성되어있습니다.

## 예시 결과
![image](https://user-images.githubusercontent.com/94027045/220042815-5b272146-a356-4511-9119-01dc2818dc6a.png)
