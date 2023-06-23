Editing...

## Model architecture
![image](https://github.com/RK-IM/ML-contents/assets/94027045/ff78664d-d2ef-4f49-967a-38a85ee18974)

The model structure is similar with GoogLeNet, which connect the 3x3 and 1x1 convolution layer parallel, but in yolo model they connected in series.

![image](https://github.com/RK-IM/ML-contents/assets/94027045/fab7687a-50c5-488d-8ea3-d143136556f1)

First layer, , 64 channels, stride: 2  
There is no information about padding size, but we can calculate from other information in convolution layer parameters.


$$[\frac{W-F+2*P}{S}] + 1$$  
$W$: Input image size  
$F$: Kernel(filter) size  
$P$: Padding size  
$S$: Stride number  

From above equation which calculate the image size after convolution layer,

For example, the first conv layer is get image size of 448x448, kernel size is 7, 64 channels and stride is 2. If input these value to equation, we get  
$[(448 - 7 + 2*P)/2] + 1 = 224$  
And the padding size of first layer is, $P = 3$.

![image](https://github.com/RK-IM/ML-contents/assets/94027045/c0880aba-6922-40ee-a40f-14c58395254b)

Activation function between all convolution layers are LeakyReLU with negative slope 0.1.

![image](https://github.com/RK-IM/ML-contents/assets/94027045/32e53e8a-0bd0-4d9a-95c4-80889cac4a9e)

![image](https://github.com/RK-IM/ML-contents/assets/94027045/e5e26484-1349-49bf-a01f-3a584a8e0eee)

## Pascel VOC Label format
![image](https://github.com/RK-IM/ML-contents/assets/94027045/426d9644-4d55-4cb3-b62b-eb1fcfdcc620)

Reference:
- https://arxiv.org/pdf/1506.02640.pdf  
- https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html
