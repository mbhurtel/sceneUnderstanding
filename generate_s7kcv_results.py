# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#############################################################################################################

# Base directory that contains the folds data and results after training using YOLOv5s
BASE_DIR = "runs/train/S7KCV_training_results"


if not os.path.exists("all_results"):
    os.mkdir("all_results")
if not os.path.exists("all_results/S7FCV_results"):
    os.mkdir("all_results/S7FCV_results")

save_dir = "all_results/S7FCV_results"

# We loop over results.csv files for each fold of data and extract Precision, Recall, mAP@0.5 and mAP@0.5:0.95
fold_results = []
for i in range(7):
    results = pd.read_csv(f"{BASE_DIR}/results_fold_{i}/results.csv")
    fold_results.append(list(results.iloc[-1, 4:8]))

# Here we plot the data using bar chart
n = 4
r = np.arange(n)
colors = ["blue", "green", "red", "pink", "cyan", "orange", "magenta"]
width = 0.12

for i, fold in enumerate(fold_results):
    plt.bar(r + (width * i), fold, color=colors[i], width=width, edgecolor="black", label=f"Fold_{i}")

plt.xticks(r + width + 0.23, ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'])
plt.xlabel("Evaluation Metrics")
plt.ylabel("Scores")
plt.legend()

plt.savefig(f"{save_dir}/S7FCV_plot.jpg", dpi=300, bbox_inches="tight")
plt.figure().clear(True)

print(f"\nSuccess! S7FCV bar plot saved at {save_dir}/S7FCV_plot.jpg")
#######################################################################################################################

# Here we generate the comparative table of the Average Precision (AP) for all classes generated using 7 folds of data
classes = ["all", "aircraft", "ambulance", "watercraft", "barrel", "briefcase", "cannon", "car", "dagger", "dog", "handgun",
          "helmet", "horse", "missile_rocket", "motorcycle", "civilian", "rifle", "tank", "truck", "soldier"]
folds_ap_data = pd.DataFrame({"Categories": classes}).set_index("Categories")

# Looping over the results for each fold of data
for i in range(7):
    results_ap = pd.read_csv(f"{BASE_DIR}/results_fold_{i}/all_classes_AP.csv")
    folds_ap_data[f"Fold_{i}"] = list(results_ap["AP_0.5"])

# Finally storing the comparative table as the csv file in S7FCV_results directory
folds_ap_data.to_csv(f"{save_dir}/folds_AP_data.csv", index=True)

print(f"Success! S7FCV bar plot saved at {save_dir}/folds_AP_data.csv")
#######################################################################################################################
