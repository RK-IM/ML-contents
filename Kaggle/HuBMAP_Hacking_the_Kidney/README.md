https://www.kaggle.com/competitions/hubmap-kidney-segmentation/overview

This will be the directories of final result. Because of large size of original tiff files, first split every file into certain size of images. If run codes in `slice_images.ipynb`, the sliced images will saved at each tiff file id folder, which is under `tiles` folder.  
`training.ipynb` in code directory runs the training and validation of model by multiple folds. For each trials, config( the parameters for training is in), best model weights(best validation loss and best dice score each), and results are saved in `logs` folder.   
```
./  
├─ code/  
│  ├─ params.py  
│  ├─ dataset.py  
│  ├─ models.py  
│  ├─ train.py  
│  ├─ main.py  
│  ├─ utils.py  
│  ├─ slice_images.ipynb  
│  └─ training.ipynb  
│  
├─ data/  
│  ├─ train/  
│  │  └─ tiff files  
│  ├─ test/  
│  │  └─ tiff files  
│  └─ tiles/  
│     ├─ image_id/  
│     │  ├─ images/  
│     │  │  └─ sliced images  
│     │  └─ masks/  
│     │     └─ sliced masks  
│     ├─ image_id/  
│     ├─ ...  
│  
└─ logs/  
   ├─ date/  
   │  ├─ best_model.pth  
   │  ├─ config.json  
   │  ├─ results.txt  
   ├─ date/  
   ├─ ...
```
