pytorch lightning에서 efficientdet 사용법입니다. EfficientDet은 [rwightman/efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)에서 가져왔습니다.
전체 과정은 Chris Hughes의 [Medium blog](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f)를 참고했습니다.
데이터도 위 블로그에서 사용한 [kaggle의 자동차 데이터셋](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)을 사용했습니다.

전체 폴더 디렉토리는 다음과 같습니다.
```
./  
├─ code/  
│  ├─ datamodules.py                         # custom pytorch lightning datamodule.  
│  ├─ dataset.py                             # pytorch dataset  
│  ├─ models.py                              # pytorch lightning module
│  ├─ params.py                              # file directory, architecture parameter
│  ├─ train.py                               # model training
│  ├─ utils.py                               # utility functions for plot prediction
│  └─ inference.ipynb                        # predict
│  
├─ data/  
│  ├─ training_images/                       # images for training
│  │  ├─ images
│  │  ├─ ... 
│  ├─ testing_images/                        # images to predict
│  │  ├─ images
│  │  ├─ ... 
│  └─ train_solution_bounding_boxes (1).csv  # csv file with boundary boxes
│  
└─ tb_logs/                                  # Log files
   └─ cars_detect/
      ├─ version_0/
      │  ├─ checkpoints/                     # trained model checkpoints
      │  │  ├─ .ckpt
      │  │  ├─ ...
      │  ├─ tfevents                         # training results
      │  └─ hparams.yaml                     # hyperparameters
      ├─ ...
```
