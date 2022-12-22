## <div align="center"> Battlefield Scene Understanding </div>

### Install the dependencies
To install all the dependencies and packages, please refer to the following code:

```
pip install -r requirements.txt
```



### Customized stratified 7-fold cross validation
If you want to create generate the 7-Folds of data, then follow the steps:
1. Download our COBA_unsplitted dataset from [here](https://github.com/9characters) and place it into the working directory.
2. Run [customized_stratified_7_fold_cv.py](https://github.com/4characters/research3/blob/main/requirements.txt) script:
```
python customized_stratified_7_fold_cv.py
```

After this step, you will get the following 7 sets of training and validation data.
```
valid_fold_0
valid_fold_1
...
valid_fold_6
```

##### Generate the plots

Notice that a new folder COBA will be created using the valid_fold_5 training and validation sets which has provided the best results. This COBA dataset will be used to train our Battlefield Obeject Detector (YOLOv5 model).

### ODM Training using COBA
You need to train 5 different YOLOv5 models independently with by running the following lines of code one by one.
```
python train.py --img 416 --batch 128 --epochs 200 --data data.yaml --cfg ./models/battlefield_yolov5n.yaml --weights '' --name results_yolov5n --cache
python train.py --img 416 --batch 64 --epochs 200 --data data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name results_yolov5s --cache
python train.py --img 416 --batch 40 --epochs 200 --data data.yaml --cfg ./models/battlefield_yolov5m.yaml --weights '' --name results_yolov5m --cache
python train.py --img 416 --batch 32 --epochs 200 --data data.yaml --cfg ./models/battlefield_yolov5l.yaml --weights '' --name results_yolov5l --cache
python train.py --img 416 --batch 16 --epochs 200 --data data.yaml --cfg ./models/battlefield_yolov5x.yaml --weights '' --name results_yolov5x --cache
```

Notice that in each independent training, the batch size is different i.e., larger the model, smaller the batch size. The YOLOv5 architectures are defined the [models](https://github.com/4characters/research3/tree/main/models) directory.

Please make sure that the output generated after training reside in the working directory. So that we can generate the comparative training plots.

The training results are stored in runs...
##### Generate training plots

### Experiment using ODM, DEM and KMC together
Now we use our trained models to make the inference on 100 new test images. Please follow the following steps:
1. Download our Battlefield Object Detector trained weights from [here](https://github.com/9characters)
2. Download the pretrained monodepth2 stereo model from [here](https://github.com/9characters) and store in [depth_models](https://github.com/4characters/research3/depth_models)
3. Download the test_images from here and place the folder into the working directory
3. Make sure you organize the models and test_images in the working directory as the following structure:
```
sceneUnderstanding
├── weights
│   ├── battlefield_object_detector.pt
├── depth_models
│   ├── stereo_model
│   │   ├── encoder.pth
│   │   ├── depth.pth
│   ├── test_images
│   │   ├── test (1).jpg
│   │   ├── test (2).jpg
│   │   ├── ...
│   │   ├── test (100).jpg
├── architectures   
├── models  
│...
```
4. Run [detect.py](https://github.com/4characters/research3/blob/main/detect.py) on the 100 test data
```
python detect.py
```

After running the above script, you will notice a new [Output]() folder that contains the output results for all 100 images. The [experimental_results]() contains the [inertia_plot](), [silhouette_scores_plot]() and [time_taken_plot]().

#### <div align="left"> Credits </div>
- Most part of the codes for Battlefield Object Detector are depicted from:
[YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)

- The depth estimation code is extracted and modified from: [Monodepth2](https://github.com/nianticlabs/monodepth2)

