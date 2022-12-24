import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.markers as markers

def generate_inertia_plot(cluster_features_csvs, save_dir):
    '''
    This function generates the inertia plot for all 100 images

    args:
    cluster_features_csvs - CSV files list that contains the distance features for all the 100 images
    '''

    print(f"\nPlotting inertia plots ...")
    
    # Initialize the list to record the list of inertia values for k in range(1,7) for all images
    global_inertia_list = []
    
    # Loop over all the csv files
    for csv in cluster_features_csvs:
        
        # List to store the inertia values for the current csv
        local_inertia_list = []
        
        # Loop 6 times to get the inertia for each value of k
        for k in range(1,7):
            
            # Read from current csv file
            data = pd.read_csv(csv)

            # We need only left-coordinate and distance features from the data
            data = data[["leftCoord", "distance"]]

            # In some cases, there may be less number of objects than the value of k, so we assign k=len(data) in such cases
            if len(data) < k:
                k = len(data)
            
            # Here, we perform the Kmeans clustering using the given parameters
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=20, tol = 1e-04, random_state = 9)
            
            # Generating the clusters
            y_km = km.fit_predict(data)

            # Store the local inertia for the current csv
            local_inertia_list.append(km.inertia_)
        
        # Finally store the local inertia list into the global list
        global_inertia_list.append(local_inertia_list)
    
    # Now we loop over the global list and plot the local inertia at different k-values
    for i, inertia in enumerate(global_inertia_list):
        
        # For reproducibility, we fix the random seed
        np.random.seed((i*25) + 469)
        color = (np.random.random(), np.random.random(), np.random.random())
        
        # PLotting the k-values and respective inertia
        plt.plot(range(1, 7), inertia, marker="o", c = color, linewidth=2, markersize=3)
        plt.xlabel("K")
        plt.ylabel("Inertia")

    plt.grid()
    plt.savefig(f"{save_dir}/inertia_plot.jpg", dpi=300, bbox_inches="tight")
    plt.figure().clear(True)
    print(f"Inertia plot successfully generated! Saved to {save_dir}/inertia_plot.jpg\n")


def plot_silhouette_scores(data_stats, save_dir):
    '''
    This function plots the silhouette scores for all 100 images

    args:
    data_stats - Data recorded by running the ODM + DEM + KMC modules
    '''

    print("Plotting silhouette scores...")
    
    # Specifying the marker for the plot
    # marker = markers.MarkerStyle(marker='o', fillstyle='none')
    image_num = range(0, len(data_stats))
    
    # Plotting the chart for 100 images with their respective silhouette scores
    plt.plot(image_num, data_stats["silhouette_score"], c = "maroon", marker = 'o')
    plt.xlabel("Images")
    plt.ylabel("Silhouette Score")
    
    # Fixing the y-limit since the silhouette score ranges from -1 to +1
    plt.ylim(-1, 1)
    plt.xlim(0, len(data_stats) - 1)
    plt.grid()
    plt.savefig(f"{save_dir}/silhouette_scores.jpg", dpi=300, bbox_inches="tight")

    print(f"Silhouette plot successfully generated! Saved to {save_dir}/silhouette_scores.jpg")
    avg_sil = np.mean(data_stats["silhouette_score"])
    plt.figure().clear(True)
    print(f"Average silhouette score is: {avg_sil}\n")


def generate_time_data(data_stats, save_dir):
    '''
    This function plots the time-taken by each module (DEM, ODM, KMC)

    args:
    data_stats - Data recorded by running the ODM + DEM + KMC modules
    '''

    print("Gnerating time-taken plot and table...")

    # Plotting ODM, DEM, KMC times
    x = range(0, len(data_stats))
    plt.plot(x, data_stats["od_time"], c = "green",  linestyle = '-', marker = "x", label="ODM")
    plt.plot(x, data_stats["depth_time"], c="maroon", linestyle = '-.', label="DEM")
    plt.plot(x, data_stats["clustering_time"], c="blue", linestyle = ':', label="KMC")
    
    # Plotting the total time
    data_stats["total_time"] = data_stats["od_time"] + data_stats["depth_time"] + data_stats["clustering_time"]
    plt.plot(x, data_stats["total_time"], c="purple", linestyle = '--', label="Total")
    
    # Adjusting the x and y limits
    plt.ylim(0, max(data_stats["total_time"]) + (0.2 * max(data_stats["total_time"])))
    plt.xlim(0, len(data_stats) - 1)
    plt.xlabel("Images")
    plt.ylabel("Time Taken by Each Alogithm (s)")

    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(f"{save_dir}/time_taken.jpg", dpi=300, bbox_inches="tight")
    plt.figure().clear(True)

    # Here we generate the table of minimum, maximum and average time taken by all the modules
    time_table = pd.DataFrame(index=["Object Detection", "Depth Estimation", "K-Means Clustering", "Total"])
    model_times = [[data_stats[model_time].min(), data_stats[model_time].max(), data_stats[model_time].mean()] for model_time in ["od_time", "depth_time", "clustering_time", "total_time"]]
    time_table["Min Time (s)"] = [model_time[0] for model_time in model_times]
    time_table["Max Time (s)"] = [model_time[1] for model_time in model_times]
    time_table["Avg Time (s)"] = [model_time[2] for model_time in model_times]
    time_table.to_csv(f"{save_dir}/time_taken_table.csv", index=True)

    print(f"Time-taken plot and table successfully generated! Saved to {save_dir}\n")