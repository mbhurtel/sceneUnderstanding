# Importing the necessary packages
import os
import pandas as pd
import numpy as np
import glob

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, cv2, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

from calculate_depth import get_depth_matrix, get_depth_value
from kmeans_clustering import create_clusters
import plots_generator as pg

@torch.no_grad()
def run(weights, source, save_dir, imgsz, conf_thresh, iou_thresh, device_keyword, depth_model, k=3):

    # Creating the directories to store the output
    if not os.path.exists("Output/detections"):
        os.mkdir(f"{save_dir}/detections")

    if not os.path.exists(f"{save_dir}/info_csv"):
        os.mkdir(f"{save_dir}/info_csv")

    if not os.path.exists(f"{save_dir}/depth_obj"):
        os.mkdir(f"{save_dir}/depth_obj")

    if not os.path.exists(f"{save_dir}/depth_maps"):
        os.mkdir(f"{save_dir}/depth_maps")

    if not os.path.exists(f"{save_dir}/clustering_plots"):
        os.mkdir(f"{save_dir}/clustering_plots")

    # Load model
    device = torch.device(device_keyword)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    # Loading the test data
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    stats = pd.DataFrame(columns=["image_id", "objects_count", "od_time", "depth_time", "clustering_time", "silhouette_score"])

    # Run the model for inference
    dt = [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Battlefield Object Detector Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Non-max suppression
        pred = non_max_suppression(pred, conf_thresh, iou_thresh, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        # Extract the image_id
        image_id = path.split(source)[-1].split(".jpg")[0].strip("\\").strip("/")

        # Get the depth matrix of the input image using the depth_model in main
        depth_matrix, depth_save_path, depth_time = get_depth_matrix(path, image_id, depth_model)

        # Dataframe to store the features for the K-Means Clustering Part
        objects_df = pd.DataFrame(columns=["object", "leftCoord", "distance"])

        # Process predictions
        det = pred[0]
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        s += '%gx%g ' % im.shape[2:]

        # Annotator to annotate the images with bounding boxes and labels
        annotator = Annotator(im0, example=str(names))
        depth_map = cv2.imread(depth_save_path)

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Here we loop to annotate and save the results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                # Annotator object to annotate the depth map
                depth_annotator = Annotator(depth_map, example=str(names))
                depth_annotator.box_label(xyxy, label, color=colors(c, True), show_conf=False)

                # Bounding box height and width
                bb_h, bb_w = c2[1] - c1[1], c2[0] - c1[0]

                # Here we get the mean depth value and the corresponding depth map annotated with center depth region
                cluster_feat_1, depth_map = get_depth_value(depth_matrix, bb_h, bb_w, depth_map, c1)

                # x-coordinate of center pixel of the predicted bounding box as second feature for clustering
                cluster_feat_2 = (c1[0] + c2[0])/2

                # Store the object names and their spatial distances in the dataframe
                objects_df = objects_df.append({"object":label.split(" ")[0], "leftCoord": cluster_feat_2,
                                                "distance": cluster_feat_1}, ignore_index=True)

            objects_df.to_csv(f"{save_dir}/info_csv/{image_id}.csv", index=False)

            if not os.path.exists(f"{save_dir}/depth_obj"):
                os.mkdir(f"{save_dir}/depth_obj")

            cv2.imwrite(f"{save_dir}/depth_obj/{image_id}.jpg", depth_map)

            # Cluster Objects
            n_obj = len(objects_df["object"])

            # We assign the value of k = 3, but in case we the ODM detects less number of objects, then we assign k to be less than 3
            k = n_obj if n_obj < 3 else (2 if n_obj == 3 else 3)

            # Create the clusters and get the results
            silhouette_score, inertia, clustering_time = create_clusters(k, objects_df, image_id, depth_matrix.shape[1])

            # Data to store in stats.csv
            obj_data_dict = {"image_id":image_id, "objects_count":n_obj, "od_time": t3-t2, "depth_time": depth_time, 
                             "clustering_time": clustering_time, "silhouette_score": silhouette_score, "clustering_inertia": inertia}

            stats = stats.append(obj_data_dict, ignore_index=True)
            im0 = annotator.result()

            cv2.imwrite(f"Output/detections/{image_id}.jpg", im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    stats.to_csv(f"Output/stats.csv", index=False)

if __name__ == "__main__":
    weights = "weights/battlefield_object_detector.pt" # Path to the weight file (*.pt)
    source = "test_images" # Folder containing test images
    imgsz = (416, 416)  # Image Size
    conf_thresh = 0.35   # Confidence Threshold
    iou_thresh = 0.45   # IoU Threshold
    device_keyword = 'cpu'   # We use CPU for the processing of all ODM, DEM and KMC modules
    k = 3 # Number of clusters

    # Create the output directory
    if not os.path.exists("Output"):
        os.mkdir("Output")
    save_dir = "Output"

    # Depth model location
    depth_model = "stereo_model"

    # This function runs the ODM + DEM + KMC modules and generates results
    run(weights, source, save_dir, imgsz, conf_thresh, iou_thresh, device_keyword, depth_model)

    if not os.path.exists("Output/experimental_plots"):
        os.mkdir("Output/experimental_plots")

    # List the csv files inside info_csv generated from the ODM + DEM + KMC module (by running detect.py)
    cluster_features_csvs = glob.glob("Output/info_csv/*.csv")

    # Here we use the inertia for each test images from stats.csv to generate the inertia plot
    pg.generate_inertia_plot(cluster_features_csvs)

    # Reading the stats file generated from the ODM + DEM + KMC module (by running detect.py)
    data_stats = pd.read_csv("Output/stats.csv")

    # Here we plot the silhouette scores for each test image from stats.csv to plot the silhouette scores
    pg.plot_silhouette_scores(data_stats)

    # Here we generate the time_taken plot for all test images
    pg.plot_time_taken(data_stats)