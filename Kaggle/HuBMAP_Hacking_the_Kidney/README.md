https://www.kaggle.com/competitions/hubmap-kidney-segmentation/overview

This will be the directories of final result. Because of large size of original tiff files, first split every file into certain size of images. If run codes in `slice_images.ipynb`, the sliced images will saved at each tiff file id folder, which is under `tiles` folder.  
`training.ipynb` in code directory runs the training and validation of model by multiple folds. For each trials, config( the parameters for training is in), best model weights(best validation loss and best dice score each), and results are saved in `logs` folder.   
```
./  
├─ code/  
│  ├─ params.py                    # Parameters for slicing images and training  
│  ├─ dataset.py                   # Pytorch dataset and Albumentation transforms  
│  ├─ models.py                    # Segmentation models pytorch  
│  ├─ train.py                     # Pytorch model Trainer
│  ├─ main.py                      # Declare dataset, dataloader, model etc.
│  ├─ utils.py                     # Encode/Decode rle, slicing, metric, save function etc.
│  ├─ slice_images.ipynb           # Notebook for slicing original tiff files to smaller tiles
│  └─ training.ipynb               # Notebook for training model with multiple folds.
│  
├─ data/  
│  ├─ train/                       # Original tiff files for training
│  │  └─ tiff files  
│  ├─ test/                        # Original tiff files for test
│  │  └─ tiff files  
│  └─ tiles/  
│     ├─ image_id/                 # Each id is from tiff file names
│     │  ├─ images/                # Sliced images and sliced masks in this directories.
│     │  │  └─ sliced images       # Same index of image and mask are pair 
│     │  └─ masks/                 # index is second part of image/mask file name.
│     │     └─ sliced masks  
│     ├─ image_id/  
│     ├─ ...  
│  
└─ logs/                           # Log files
   ├─ date/
   │  ├─ experience_0/
   │  │  ├─ best_model.pth         # Trained model weights with
   │  │  │                         # best validation loss and dice score for each fold
   │  │  ├─ config.json            # parameters for training
   │  │  └─ results.txt            # results of training. fold, epoch, phase, loss, dice are saved.
   │  ├─ experience_1/
   │  ├─ ...
   ├─ date/  
   ├─ ...
```

![image](https://github.com/RK-IM/ML-contents/assets/94027045/9a790105-ff95-4a66-ae62-06435ea81ff0)
