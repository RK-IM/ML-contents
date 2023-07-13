## Model architecture
![image](https://github.com/RK-IM/ML-contents/assets/94027045/6fb3414d-c982-4b12-a710-6f44a6d73f03)  

The model structure is similar with GoogLeNet, which connect the 3x3 and 1x1 convolution layer parallel, but in yolo v1 model they connected in series.  

![image](https://github.com/RK-IM/ML-contents/assets/94027045/9a9583ed-9b76-41a3-9582-f74624b8c951) 

First layer has 64 channels with 7x7 kernel size and stride is 2.  
There is no information about padding size, but we can calculate from other information in convolution layer parameters.

$$
[\frac{W-F+2*P}{S}] + 1
$$

$W$: Input image size  
$F$: Kernel(filter) size  
$P$: Padding size  
$S$: Stride number  

From above equation which calculate the image size after convolution layer,

For example, the first conv layer is get image size of 448x448, kernel size is 7, 64 channels and stride is 2. If input these value to equation, we get  
$[(448 - 7 + 2*P)/2] + 1 = 224$  
And the padding size of first layer is, $P = 3$.

![image](https://github.com/RK-IM/ML-contents/assets/94027045/42a4472c-fe06-42e9-b07d-be8a399530bd)  

Activation function between all convolution layers are LeakyReLU with negative slope 0.1.  

![image](https://github.com/RK-IM/ML-contents/assets/94027045/84049467-4103-47cf-9841-487b524c44ff)  

And add dropout layer between fully connected layers.

![image](https://github.com/RK-IM/ML-contents/assets/94027045/498c0476-8378-4f49-b1ba-33f0c613cad5)  

The outputs shape will be [batch_size, 1470] that 1470 equals to  $S * S * (B * 5 + C)$ when S=7, B=2, C=20. For each cell, model predict two boundary boxes(B=2) and Pascal VOC has 20 class. So the value of each tensor is, class probability to 0\~19th index and confidence(20th, 25th), boundary box coordinates(21st\~24th, 26~29th) for each boxes.

## Data
The training data is [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/). This dataset has 20 classes(examples [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/examples/index.html)) and labels are stored in xml file. xml file has same name as related image file, and each file has information like below image. Boundary boxes represent coordinates with corner values.

![image](https://github.com/RK-IM/ML-contents/assets/94027045/4ee238f2-e35f-4cf5-8110-21583c82a747)  

## Loss

![image](https://github.com/RK-IM/ML-contents/assets/94027045/fa0c0ece-b509-4ace-bc00-32b6b72dc12c)  

They use Sum-squared error to A) center of boundary box, B) width and height of boundary box, C) confidence, D) box which does not contain objects, E) class probability. Width and height error could be large compare to other loss and leads to suppress other losses, square root the value before calculate. And parameter $\lambda_{coord}$ and $\lambda_{noobj}$ are multiplied at A, B and D error. Because as written on paper, lots of cell doesn't contain any objects and this could overpowering the gradient of cell which has objects.  

## Metric
- Intersection over Union
- Non-Maximum Suppression
- mean Average Precision

## Train

![image](https://github.com/RK-IM/ML-contents/assets/94027045/cc774b51-ed0d-44c3-b1b4-755946de32a2)  

The model has trained for 135 epochs with batch size of 64. Use SGD optimizer, momentum of 0.9 and weight decay of 0.0005. Learning schedule is rasing $10^{-3}$ to $10^{-2}$ and keep training with $10^{-2}$ for 75 epochs, $10^{-3}$ for 30 epochs, and another 30 epochs with $10^{-4}$.

However, this implementation, uses the `Adam` optimizer with learning rate of 2e-5, as SGD with momentum somtimes gradient explodes or learns too slowly. So learning rate increases from 1/10 of 2e-5 to 2e-5 and decreases linearly every iteration.  
Limitation of resources and time, I use `Stratified Group KFold` from the sklearn library to split the VOC2012 dataset into `10 folds`. And then use only `one fold`, split again 80% for training and 20% for validation, also using Stratified Group KFold. The reason for using stratified group kfold is that the boundary boxes on same image must be in the same fold and the object classes are not balanced. Using this subset, training for `100` epochs.  

The total time to train the Yolo v1 model using these parameters and the RTX 3060 was about 2 hours. The training and validation steps only take 40~50 seconds but the mAP calculation takes about 10 seconds. Since it is an early model of yolo, it is unavoidable that the speed is slow, but it can be optimized for taking a long time to calculate mAP.  

## Results

![image](https://github.com/RK-IM/ML-contents/assets/94027045/9c445c0e-b537-402e-8bf2-962c96e65c5a)  

The validation loss and mAP are much better than training. This may be due to data imbalance. If you look at the classes of objects included in the images, there is a class that contains only a few tens of thousands of photos, with 'people' class occupying the largest part. The effect must have been greater because it was trained with only a small amount of data. Even if you look at the output example, the position is matched to some extent, but the class is predicted as a person.  

![image](https://github.com/RK-IM/ML-contents/assets/94027045/6a9438a1-252d-4f5a-a719-12fb8ae2aec8)


Reference:
- https://arxiv.org/pdf/1506.02640.pdf  
- https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html
