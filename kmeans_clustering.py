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
    '''
    This function creates the clusters for the objects with distances measures

    args:
    k - Total number of clusters to be generated (3 in our case)
    data - The data with clustering features "distance" and "left coordinates"
    image_id - ID of the image to store the results
    x_dim - Width of the input image such that we can get the accurate cluster visualization

    returns:
    sil_score - Silhouette score for the given image cluster
    inertia - Inertia for the given image cluster
    clustering_time - Time taken to complete the clustering
    '''

    data = data[["leftCoord", "distance"]]

    # The x-coordinates are pretty high, and distances are lower,
    # so we scale up the distance by 100 units, such that the clustering plot won't shrink
    # We will scale to regular distance during the plotting-time
    data["distance"] *= 100
    cluster_time_1 = time_sync()

    # Running the KMeans module using the given parameters
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=20, tol = 1e-04, random_state = 9)
    y_km = km.fit_predict(data)
    cluster_time_2 = time_sync()
    clustering_time = cluster_time_2 - cluster_time_1

    # In case if number of objects < k, then silhouette score may not be computed, so we assign it to be 0.5 in such cases
    # However, our test examples do not have such cases!
    try:
        sil_score = silhouette_score(data, km.labels_, metric='manhattan')
    except:
        print("Seems like we didn't meet the silhouette criteria. Assigning sil_score = 0.5")
        sil_score = 0.5

    # Inertia of the kmeans clustering
    inertia = km.inertia_

    # Specifying the colors and markers
    colors = ["green", "red", "blue"]
    markers = ["s", "o", "v"]

    # Here we again scale down the distance
    data["distance"] /= 100

    # Plotting the datapoints and marking the multiple clusters with different markers
    for i in range(k):
        plt.scatter(data[y_km == i]["leftCoord"], data[y_km == i]["distance"], s=50, 
                    c = colors.pop(0), marker = markers.pop(0), label=f"cluster {i+1}")

    # Plotting the centroids of the clusters and marking with a '*'
    cluster_centers = np.array([[x, dist/100] for x, dist in km.cluster_centers_])
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=50, marker='*', 
                c='orange', edgecolor='orange', label='centroids')

    # Specifying the scatter plot parameters
    fs = 18
    plt.xlabel("Center Pixel x-coordinate (px)", {"fontsize": fs})
    plt.ylabel("Absolute Distance (m)", {"fontsize": fs})
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlim(0, x_dim)
    plt.ylim(0)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(f"Output/clustering_plots/{image_id}_cluster.jpg", dpi=300, bbox_inches="tight")
    plt.figure().clear(True)

    return sil_score, inertia, clustering_time