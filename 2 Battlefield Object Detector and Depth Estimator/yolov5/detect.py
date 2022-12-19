import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, cv2, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

from calculate_depth import get_depth_matrix
from kmeans_clustering import create_clusters

# def get_second_min(all_rgn_depths):
#     temp_list = [(i, rgn) for i, rgn in enumerate(all_rgn_depths)]
#     sorted_list = sorted(temp_list, key=lambda x: x[1])
#     second_min = sorted_list[1]
#     return second_min

def get_depth_value(depth_matrix, h, w, depth_map, c1):
    # region proposals
    # coords_yyxx = [(c1[1] + int(0.25 * h), c1[1] + int(0.30 * h), c1[0] + int(0.48 * w), c1[0] + int(0.53 * w)),
    #                 (c1[1] + int(0.48 * h), c1[1] + int(0.53 * h), c1[0] + int(0.25 * w), c1[0] + int(0.30 * w)),
    #                 (c1[1] + int(0.75 * h), c1[1] + int(0.8 * h), c1[0] + int(0.48 * w), c1[0] + int(0.53 * w)),
    #                 (c1[1] + int(0.48 * h), c1[1] + int(0.53 * h), c1[0] + int(0.75 * w), c1[0] + int(0.8 * w)),
    #                 (c1[1] + int(0.48 * h), c1[1] + int(0.53 * h), c1[0] +int(0.48 * w), c1[0] + int(0.53 * w))]

    coords_yyxx = [(c1[1] + int(0.48 * h), c1[1] + int(0.53 * h), c1[0] +int(0.48 * w), c1[0] + int(0.53 * w))]

    # all_rgn_depths = [depth_matrix[y1: y2, x1: x2].mean() for y1, y2, x1, x2 in coords_yyxx]
    abs_center_depth = [depth_matrix[y1: y2, x1: x2].mean() for y1, y2, x1, x2 in coords_yyxx]

    # second_min_idx, second_min_val = get_second_min(all_rgn_depths)
    for i, coords in enumerate(coords_yyxx):
        # color = (0, 0, 255)
        # if i == second_min_idx:
        #     color = (0, 255, 0)
        cv2.rectangle(depth_map, (coords[2], coords[0]), (coords[3], coords[1]), color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

    # return second_min_val, depth_map
    return abs_center_depth[0], depth_map

@torch.no_grad()
def run(weights, source, save_dir, imgsz, conf_thresh, iou_thresh, device_keyword, depth_model, k=3):

    # Load model
    device = torch.device(device_keyword)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    time_taken = pd.DataFrame(columns=["image_id", "objects_count", "od_time", "depth_time", "clustering_time", "silhouette_score"])

    # Run inference
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

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thresh, iou_thresh, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        image_id = path.split("\\")[-1].split(".csv")[0]

        # Get the depth matrix of the input image using the depth_model in main
        depth_matrix, depth_save_path, depth_time = get_depth_matrix(path, image_id, depth_model)

        # Dataframe to store the features for the K-Means Clustering Part
        objects_df = pd.DataFrame(columns=["object", "leftCoord", "topCoord", "distance"])

        # Process predictions
        det = pred[0]
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        s += '%gx%g ' % im.shape[2:]
        annotator = Annotator(im0, line_width=18, example=str(names))
        depth_map = cv2.imread(depth_save_path)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                # Annotator object to annotate the depth map
                depth_annotator = Annotator(depth_map, line_width=18, example=str(names))
                depth_annotator.box_label(xyxy, label, color=colors(c, True), show_conf=False)

                bb_h, bb_w = c2[1] - c1[1], c2[0] - c1[0]
                xx = c1[0] + int(0.45 * bb_w), c1[0] + int(0.55 * bb_w)
                yy = c1[1] + int(0.45 * bb_h), c1[1] + int(0.55 * bb_h)

                # Drawing the center depth region where we take the mean
                # cv2.rectangle(depth_map, (xx[0], yy[0]), (xx[1], yy[1]), (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

                image_depth_crop = depth_matrix[yy[0]: yy[1], xx[0]: xx[1]]
                # cluster_feat_1 = np.mean(image_depth_crop)

                cluster_feat_1, depth_map = get_depth_value(depth_matrix, bb_h, bb_w, depth_map, c1)
                cluster_feat_2 = (c1[0] + c2[0])/2  # x-coordinate of center pixel of the predicted bounding box
                cluster_feat_3 = (c1[1] + c2[1])/2  # y-coordinate of center pixel of the predicted bounding box

                person_vehicles = ["soldier", "civilian", "tank", "truck", "cannon", "rocket_missile",
                                    "watercraft", "airplane", "ambulance", "motorcycle", "car"]

                # if label.split(" ")[0] in person_vehicles:
                #     objects_df = objects_df.append({"object":label, "leftCoord": cluster_feat_2, "distance": cluster_feat_1}, ignore_index=True)
                objects_df = objects_df.append({"object":label.split(" ")[0], "leftCoord": cluster_feat_2,
                                                "topCoord":cluster_feat_3, "distance": cluster_feat_1}, ignore_index=True)

            if not os.path.exists("Output/info_csv"):
                os.mkdir("Output/info_csv")

            csv_save_path = f"Output/info_csv/{image_id.split('.')[0]}.csv"
            objects_df.to_csv(csv_save_path, index=False)

            depth_obj_save_path = "Output/depth_obj"
            if not os.path.exists(depth_obj_save_path):
                os.mkdir(depth_obj_save_path)

            cv2.imwrite(f"{depth_obj_save_path}/{image_id.split('.')[0]}.png", depth_map)

            # Cluster Objects
            n_obj = len(objects_df["object"])

            # k = 1 if n_obj <=3 else (2 if (n_obj > 3 and n_obj <=6) else 3)
            # k = 1 if n_obj == 1 else (2 if (n_obj == 2 and n_obj <=6) else 3)
            k = n_obj if n_obj < 3 else (2 if n_obj == 3 else 3)

            silhouette_score, inertia, clustering_time = create_clusters(k, objects_df, image_id, depth_matrix.shape[1])

            obj_data_dict = {"image_id":image_id.split('.')[0], "objects_count":n_obj, "od_time": t3-t2, "depth_time": depth_time, 
                             "clustering_time": clustering_time, "silhouette_score": silhouette_score, "clustering_inertia": inertia}

            time_taken = time_taken.append(obj_data_dict, ignore_index=True)
            im0 = annotator.result()

            if not os.path.exists("Output/detections"):
                os.mkdir("Output/detections")

            cv2.imwrite(f"Output/detections/{image_id.split('.')[0]}.jpg", im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    time_taken.to_csv(f"Output/stats.csv", index=False)

if __name__ == "__main__":
    weights = "weights/best_yolov5x.pt" # Path to the weight file (*.pt)
    source = "test" # Folder containing test images
    imgsz = (416, 416)  # Image Size
    conf_thresh = 0.35   # Confidence Threshold
    iou_thresh = 0.45   # Threshold of IoU
    device_keyword = 'cpu'   # Device cpu or cuda:<n>
    k = 3 # Number of clusters

    if not os.path.exists("Output"):
        os.mkdir("Output")
    save_dir = "Output" # Save directory

    # depth_model = "mono_640x192"
    # depth_model = "mono+stereo_640x192"
    depth_model = "stereo_640x192"

    run(weights, source, save_dir, imgsz, conf_thresh, iou_thresh, device_keyword, depth_model)
