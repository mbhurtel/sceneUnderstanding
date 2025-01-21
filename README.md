## <div align="center"> Battlefield Scene Understanding </div>

![](assets/qualitative_results.bmp)

This is the official GitHub repository for the paper <b>"[Toward Scene Understanding with Depth and Object-Aware Clustering in Contested Environment](https://ieeexplore.ieee.org/abstract/document/10459876/)"</b>. We have tried our best to keep the <b>random_state (or seed)</b> where required, for reproducibility of the results. The code is organized in four broad sections:
- Environment and dependencies setup
- Customized stratified 7-fold cross validation
- ODM Training using COBA
- Inference and Experiment using ODM + DEM + KMC

*Note that: We used Google Colab P100 GPU and 16GB RAM to train our Object Detection Model, and we used CPU for the model inference assuming the resource-constraint battlefield robot.*

---

### 1. Environment and dependencies setup
#### Step 1: Virtual Environment
Create the virtual environment using <b>conda</b> or <b>virtualenv</b>. We used <b>python3.7</b> throughout the project.
``` shell
conda create --name sceneUnderstanding python=3.7
conda activate sceneUnderstanding
```

#### Step 2: Install
To install all the dependencies and packages, please execute [requirements.txt](https://github.com/9characters/sceneUnderstanding/blob/main/requirements.txt):

``` shell
cd sceneUnderstanding
pip install -r requirements.txt
```

### 2. Customized stratified 7-fold cross-validation

*Note that:*
* *If you want to perform the main training directly, head to section 3. We have provided the COBA dataset ready to train the YOLOv5 models.*
* *If you want to run the inference and clustering experiments directly, head to section 4. We have provided the trained weights of YOLOv5x and the DEM for the inference.*

This section executes our algorithm 1 presented in the paper showing how we created the stratified labels, split the COBA dataset into 7 folds and evaluated the data folds.

#### Step 1: Split the COBA dataset 7 folds of training and validation sets
- Download our COBA dataset from <a href=#>here</a> and place it into the working directory.
- Run [customized_stratified_7_fold_cv.py](https://github.com/9characters/sceneUnderstanding/blob/main/customized_stratified_7_fold_cv.py) script:

``` py
python customized_stratified_7_fold_cv.py
```

After this step, a new directory named <b>COBA_7fold_split_sets</b> is created that stores the 7 folds of the dataset organized as follows:

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

*Note that: Each fold contains 6000 training images and 1000 validation images along with their corresponding labels.*

#### Step 2: Train the YOLOv5s model on our 7 folds of dataset
We train YOLOv5s using our COBA dataset with <b>batch_size = 64</b>, <b>epochs=50</b>, and <b>image_resolution=416 x 416</b>. Please run the following code independently from command line (staying in the working directory):

``` py
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_0_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_0 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_1_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_1 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_2_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_2 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_3_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_3 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_4_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_4 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_5_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_5 --cache
python train.py --img 416 --batch 64 --epochs 50 --data data/fold_6_data.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name S7KCV_training_results/results_fold_6 --cache
```

After the completion of 7 independent training, the training results are stored in the new <b>runs/train/S7KCV_training_results</b> directory.

#### Step 3: Generate comparative plots for 7 folds of data

* Here we generate the comparative bar chart and table to find out which fold of data is optimal. To generate the results, you should simply run [generate_s7kcv_results.py](https://github.com/9characters/sceneUnderstanding/blob/main/generate_s7kcv_results.py) script.

``` py
python generate_s7kcv_results.py
```

* The results are stored in <b>all_results/S7KCV_results</b> directory. After results are generated, the structure of all_results will look like this:

```
all_results
├── S7FCV_results
      ├── folds_AP_data.csv
      ├── s7fcv_plot.jpg
```

### 3. ODM Training using COBA

#### Step 1: Training YOLOv5 models using our COBA dataset
* We need to train 5 different YOLOv5 models independently by running the following lines of code one by one.
* 
```
python train.py --img 416 --batch 128 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5n.yaml --weights '' --name results_YOLOv5n --cache
python train.py --img 416 --batch 64 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5s.yaml --weights '' --name results_YOLOv5s --cache
python train.py --img 416 --batch 40 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5m.yaml --weights '' --name results_YOLOv5m --cache
python train.py --img 416 --batch 32 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5l.yaml --weights '' --name results_YOLOv5l --cache
python train.py --img 416 --batch 16 --epochs 200 --data data/data_COBA.yaml --cfg ./models/battlefield_yolov5x.yaml --weights '' --name results_YOLOv5x --cache
```

* In each independent training, the batch size is different i.e., the larger the model, the smaller the batch size. The battlefield object detection architectures are defined in the [models](https://github.com/9characters/sceneUnderstanding/tree/main/models) directory.

* Notice that a new directory <b>runs</b> is created after the completion of the training, which stores the results for each of the independent training executions.

* Please note that the training requires a lot of time. The following table shows the time taken for us to train each model on <b>Google Colab's P100</b> GPU trained for <b>200 epochs</b>:

| Models | Training Time (h)|
|  :-:   |        :-:       |
|YOLOv5n |       1.385      |
|YOLOv5s |       2.349      |
|YOLOv5m |       4.252      |
|YOLOv5l |       7.167      |
|YOLOv5x |       13.556     |

#### Step 2: Generate training results

* Now we will use the results for all 5 YOLOv5 models i.e., YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l and YOLov5x and generate the training plots. To generate the training plots, you can simply run [training_plots_generator.py](https://github.com/9characters/sceneUnderstanding/blob/main/training_plots_generator.py).

``` py
python training_plots_generator.py
```

* The training results are stored in <b>all_results/training_results</b> directory. And the structure of the <b>all_results</b> directory will look like this:

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


### 4. Inference and Experiment using ODM + DEM + KMC
#### Step 1: Download the necessary models and data

- Download our Battlefield Object Detector trained weights (battlefield_object_detector.pt) from <a href=#>here</a> and store it in [Weights](https://github.com/9characters/research3/tree/main/weights) directory.
- Download the test_images from <a href=#>here</a> and place the folder into the working directory

Note that the pretrained monodepth2 model and associated architectures are already uploaded in the <b>depth_models/stereo</b> and <b>architectures</b> directories respectively.

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

After running the above script, the results from the inference are stored in <b>all_results/inference_output</b>, and the experimental results are stored in <b>all_results/experimental_results</b>. The structure of <b>all_output</b> directory after this step will look like this:

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

The qualitative results presented in the paper are generated inside <b>all_results/inference_output</b>.

#### 5. Credits
* The code for Battlefield Object Detector is based on
[YOLOv5](https://github.com/ultralytics/yolov5)

* The depth estimation model is extracted from: [Monodepth2](https://github.com/nianticlabs/monodepth2)

#### 6. Citation
If you find this work useful, please consider citing our work:
<pre>
@inproceedings{bhurtel2023toward,
author={Bhurtel, Manish and Siwakoti, Yuba R. and Rawat, Danda B. and Sadler, Brian M. and Fossaceca, John M. and Rice, Daniel O.},
booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)},
title={Toward Scene Understanding with Depth and Object-Aware Clustering in Contested Environment},
year={2023},
pages={1418-1425},
organization={IEEE},
doi={<a href="https://ieeexplore.ieee.org/abstract/document/10459876">10.1109/ICMLA58977.2023.00214</a>}}
</pre>


