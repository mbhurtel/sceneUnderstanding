## <div align="center"> Battlefield Scene Understanding </div>

![](assets/qualitative_results.bmp)

This is the official GitHub repository for the paper <b>"Efficient and Non-Redundant Objects Allocation Using Object-Aware and Depth-Aware Clustering for Battlefield Scenario"</b>. We have tried our best to keep the <b>random_state (or seed)</b> where required, for reproducibility of the results. The code is organized in four broad sections:
- Environment and dependencies setup
- Customized stratified 7-fold cross validation
- ODM Training using COBA
- Inference and Experiment using ODM + DEM + KMC

### Environment and dependencies setup
#### Step 1: Virtual Environment
Create the virtual environment using <b>conda</b> or <b>virtualenv</b>. We used <b>python3.7</b> throughout the project.
```
conda create --name sceneUnderstanding python=3.7
conda activate sceneUnderstanding
```

#### Step 2: Install
To install all the dependencies and packages, please execute [requirements.txt](https://github.com/9characters/sceneUnderstanding/blob/main/requirements.txt):

```
pip install -r requirements.txt
```

### Customized stratified 7-fold cross validation
##### Step 1: Split the COBA dataset 7 folds of training and validation sets
- Download our COBA dataset from <a href=#>here</a> and place it into the working directory.
- Run [customized_stratified_7_fold_cv.py](https://github.com/9characters/sceneUnderstanding/blob/main/customized_stratified_7_fold_cv.py) script:
```
python customized_stratified_7_fold_cv.py
```

After this step, you will get the following 7 sets of training and validation data in a new directory named <b>COBA_7fold_split_sets</b> organized as follows:

```
COBA_7fold_split_sets
├── fold_0
│     ├── train
│     ├── valid
├── fold_1
│     ├── train
│     ├── valid
│     ...
├── fold_6
      ├── train
      ├── valid
```

#### Step 2: Train the YOLOv5s model on our 7 folds of dataset
We train YOLOv5s using our COBA dataset with <b>batch_size = 64</b>, <b>epochs=50</b>, and <b>image_resolution=416 x 416</b>. Please run the following code independently from commandline (staying in the working directory):
```
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_0_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_0 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_1_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_1 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_2_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_2 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_3_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_3 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_4_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_4 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_5_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_5 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_6_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_6 --cache
```

After completion of 7 independent training, the training results are stored in the a new <b>runs/train/S7KCV_training_results</b> directory. These results are used to generate the comparative plots.

##### Step 3: Generate comparative plots for 7 folds of data
Here we generate the comparative bar chart and table to find out which fold of data is optimal. To generate the results, you should simply run [generate_s7kcv_results.py](https://github.com/9characters/sceneUnderstanding/blob/main/generate_s7kcv_results.py) script.
```
python generate_s7kcv_results.py
```

The results are stored in <b>all_results/S7KCV_results</b> directory. After results are generated, the structure of all_results will look like this:

```
all_results
├── S7FCV_results
      ├── folds_AP_data.csv
      ├── s7fcv_plot.jpg
```

### ODM Training using COBA
#### Step 1: Training YOLOv5 models using our COBA dataset
We need to train 5 different YOLOv5 models independently by running the following lines of code one by one.
```
python train.py --img 416 --batch 128 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5n.yaml --weights '' --name results_YOLOv5n --cache
python train.py --img 416 --batch 64 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name results_YOLOv5s --cache
python train.py --img 416 --batch 40 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5m.yaml --weights '' --name results_YOLOv5m --cache
python train.py --img 416 --batch 32 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5l.yaml --weights '' --name results_YOLOv5l --cache
python train.py --img 416 --batch 16 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5x.yaml --weights '' --name results_YOLOv5x --cache
```

In each independent training, the batch size is different i.e., larger the model, smaller the batch size. The battlefield object detection architectures are defined in the [models](https://github.com/9characters/sceneUnderstanding/tree/main/models) directory.

Notice that a new directory <b>runs</b> is created after the completion of the training, that stores the results for each of the independent training execution.

Please note that the training requires a lot of time. Following table shows the time taken for us to train each model on <b>Google Colab's P100</b> GPU trained for <b>200 epochs</b>:

| Models | Training Time (h)|
|  :-:   |        :-:       |
|YOLOv5n |       1.385      |
|YOLOv5s |       2.349      |
|YOLOv5m |       4.252      |
|YOLOv5l |       7.167      |
|YOLOv5x |       13.556     |

#### Step 2: Generate training results
Now we will use the results for all 5 YOLOv5 models i.e., YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l and YOLov5x and generate the training plots. To generate the training plots, you can simply run [training_plots_generator.py](https://github.com/9characters/sceneUnderstanding/blob/main/training_plots_generator.py).
```
python training_plots_generator.py
```
The results are stored in <b>all_results/training_results</b> directory. And the structure of the <b>all_results</b> directory will look like this:

```
all_results
├── S7FCV_results
│     ├── folds_AP_data.csv
│     ├── s7fcv_plot.jpg
├── training_results
      ├── results.jpg
      ├── PR_curve.png
      ├── results_table.csv
```


### Inference and Experiment using ODM + DEM + KMC
#### Step 1: Download necessary models and data

- Download our Battlefield Object Detector trained weights (battlefield_object_detector.pt) from <a href=#>here</a> and store it in [Weights](https://github.com/9characters/research3/tree/main/weights) directory.
- Download the test_images from <a href=#>here</a> and place the folder into the working directory

Note that the pretrained monodepth2 model and associated architectures is already uploaded in the <b>depth_models/stereo</b> and <b>architectures</b> directories respectively.

Make sure you organize the models and test_images in the working directory as the following structure:
```
sceneUnderstanding
│...
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
│   ├── depth_decoder.py
│   ├── resnet_encoder.py
├── models
│...
```
Now simply run [detect.py](https://github.com/9characters/sceneUnderstanding/blob/main/detect.py) to run the inference using (ODM + DEM + KMC) on 100 test_images.
```
python detect.py
```

After running the above script, the results from the inference are stored in <b>all_results/inference_output</b> and the experimental results are stored in <b>all_results/experimental_results</b>. The structure of <b>all_output</b> directory after this step will look like this:

```
all_results
├── S7FCV_results
│     ├── folds_AP_data.csv
│     ├── s7fcv_plot.jpg
├── training_results
│     ├── results.jpg
│     ├── PR_curve.png
│     ├── results_table.csv
├── inference_output
│     ├── detections
│     ├── depth_maps
│     ├── depth_obj
│     ├── clustering_plots
│     ├── info_csv
│     ├── stats.csv
├── experimental_results
      ├── inertia_plot.jpg
      ├── silhouette_scores.jpg
      ├── time_taken.jpg
```
#### <div align="left"> Credits </div>
- The code for Battlefield Object Detector is based on
[YOLOv5](https://github.com/ultralytics/yolov5)

- The depth estimation model is extracted from: [Monodepth2](https://github.com/nianticlabs/monodepth2)

