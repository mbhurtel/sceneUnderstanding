import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.torch_utils import time_sync
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def create_clusters(k, data, image_id, x_dim):

    if not os.path.exists("Output/clustering_plots"):
        os.mkdir("Output/clustering_plots")

    data = data[["leftCoord", "distance"]]

    # Since the x-coordinates are pretty high, and distances can be low, we multiply the distances by 100, to maintain the balance 
    # We do not have the real-world scale of the pixel coordinates system of the image
    # We choose 100 since the test images because most of the images have the 4k resolution

    data["distance"] *= 100

    cluster_time_1 = time_sync()
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=20, tol = 1e-04, random_state = 9)
    y_km = km.fit_predict(data)
    cluster_time_2 = time_sync()
    clustering_time = cluster_time_2 - cluster_time_1

    try:
        sil_score = silhouette_score(data, km.labels_, metric='manhattan')
    except:
        print("Seems like we didn't meet the silhouette criteria. Assigning sil_score = 0.5")
        sil_score = 0.5
    
    inertia = km.inertia_

    colors = ["green", "red", "blue"]
    markers = ["s", "o", "v"]

    # To plot the values, we again reduce the the distance values to normal scale by dividing by 100
    data["distance"] /= 100

    for i in range(k):
        plt.scatter(data[y_km == i]["leftCoord"], data[y_km == i]["distance"], s=50, 
                    c = colors.pop(0), marker = markers.pop(0), label=f"cluster {i+1}")

    cluster_centers = np.array([[x, dist/100] for x, dist in km.cluster_centers_])
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=50, marker='*', 
                c='orange', edgecolor='orange', label='centroids')

    fs = 18
    plt.xlabel("Center Pixel x-coordinate (px)", {"fontsize": fs})
    plt.ylabel("Absolute Distance (m)", {"fontsize": fs})
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlim(0, x_dim)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(f"Output/clustering_plots/{image_id}_cluster.jpg", dpi=300, bbox_inches="tight")
    plt.figure().clear(True)

    return sil_score, inertia, clustering_time