https://www.kaggle.com/competitions/hubmap-kidney-segmentation/overview
This competition was detecting glomeruli FTU. The datasets are 15 tiff files, one csv file which has rle labels for each tiff file, and 5 test tiff files. The target metric is dice score.

![preview](https://github.com/RK-IM/ML-contents/assets/94027045/acc25ef6-fadb-42da-b5c2-a1fe5ef83e4f)

I chose a small model because of resource limitations and training time. Cropped the original tiff file to 1024x1024 and then reduced it by a factor of 4 to use a 256x256 image for training. The total number of splited images is about 12000, and it took 6 hours and 30 minutes to learn 5 folds with the rtx3060.

Model info  
- Model structure: Unet  
- Backbone model: EfficientNet B1  
- Backbone pretrained weight: noisy-student  

This will be the directories of final result. If run codes in `slice_images.ipynb`, the sliced images will saved at each tiff file id folder, which is under `tiles` folder.  
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
