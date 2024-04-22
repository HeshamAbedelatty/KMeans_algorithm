from tkinter import *
from tkinter import filedialog
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from random import sample


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_entry.delete(0, END)
    file_entry.insert(END, file_path)


def start_analysis():
    file_path = file_entry.get()
    percentage = float(percentage_entry.get())
    k = int(k_entry.get())

    # Calculate the total number of rows in the file
    total_rows = sum(1 for _ in open(file_path))

    # Calculate the number of rows to read
    nrows_to_read = int(total_rows * (percentage / 100))

    # Read only the 'IMDB Rating' column from the CSV file
    data = pd.read_csv(file_path, usecols=['Movie Name', 'IMDB Rating', 'Votes'], nrows=nrows_to_read)
    movie_data, outliers, u, l = find_outliers(data)
    rates_data = list(movie_data['IMDB Rating'])

    # Initialize centroids randomly
    # centroids = initialize_centroids(k, rates_data)

    centroids, clusters = kmeans_clustering(rates_data, k)

    # Display clusters
    clusters_text.config(state=NORMAL)
    clusters_text.delete(1.0, END)
    clusters_text.insert(END, "Clusters:\n")
    for key, values in clusters.items():

        clusters_text.insert(END, f"------------------------------------------------------------------------------")
        clusters_text.insert(END, f"\n\n\nCentroid: {key}\n")
        clusters_text.insert(END, f"------------------------------------------------------------------------------")
        clusters_text.insert(END, f"--number of item: {len(values)}-------------")
        clusters_text.insert(END, f"------------------------------------------------------------------------------")

        for value in set(values):
            # clusters_text.insert(END, f"{value}\n")
            v = movie_data[movie_data['IMDB Rating'] == value]

            clusters_text.insert(END, f"{v}\n")

    # Display centroids
    centroids_text.config(state=NORMAL)
    centroids_text.delete(1.0, END)
    centroids_text.insert(END, "Centroids:\n")
    centroids_text.insert(END, centroids)

    # Display upper bound and lower bound
    centroids_text.insert(END, "\n\nUpper bound:\n")
    centroids_text.insert(END, u)
    centroids_text.insert(END, "\nlower bound:\n")
    centroids_text.insert(END, l)
    # Detect outliers

    # Display outliersoutliers_text
    outliers_text.config(state=NORMAL)
    outliers_text.delete(1.0, END)
    outliers_text.insert(END, f"---------number of of outliers: {len(outliers)}-------------")
    outliers_text.insert(END, "\n Outliers:\n")
    for line in outliers:
        # v = data[data['IMDB Rating'] == line]
        # outliers_text.insert(END, f"{v}\n")
        outliers_text.insert(END, f"{line}\n")

    outliers_text.config(state=DISABLED)


def find_outliers(movies_cleaned):
    x1 = movies_cleaned['IMDB Rating'].quantile(0.25)
    x3 = movies_cleaned['IMDB Rating'].quantile(0.75)
    # print(x1)
    # print(x3)
    IQR = x3 - x1
    l = x1 - 1.5 * IQR  # lower
    u = x3 + 1.5 * IQR  # upper
    # print(l)
    # print(u)

    outliers = movies_cleaned[(movies_cleaned['IMDB Rating'] < l) | (movies_cleaned['IMDB Rating'] > u)]

    movies_cleaned = movies_cleaned[(movies_cleaned['IMDB Rating'] >= l) & (movies_cleaned['IMDB Rating'] <= u)]

    return movies_cleaned, outliers['IMDB Rating'], u, l


def initialize_centroids(c, data):
    while True:
        centroids = sample(list(data), c)
        if len(centroids) == len(set(centroids)):
            break
    return centroids


def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def assign_to_clusters(data, centroids):
    clusters = {}
    for point in data:
        min_distance = float('inf')
        closest_centroid = None
        for centroid in centroids:
            distance = euclidean_distance(point, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_centroid = centroid
        if closest_centroid in clusters:
            clusters[closest_centroid].append(point)
        else:
            clusters[closest_centroid] = [point]
    return clusters


def update_centroids(clusters):
    centroids = []
    for centroid, cluster_points in clusters.items():
        mean_point = np.mean(cluster_points)
        # mean_point = round(mean_point, 1)
        centroids.append(mean_point)
    return centroids


def kmeans_clustering(data, k):
    centroids = initialize_centroids(k, data)

    while True:
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids, clusters


# def detect_outliers(clusters):
#     outliers = {}
#     for cluster_centroid, cluster_points in clusters.items():
#         mean = np.mean(cluster_points)
#         std = np.std(cluster_points)
#         outlier_threshold = mean + int(std_multiplier_entry.get()) * std
#         cluster_outliers = [point for point in cluster_points if point > outlier_threshold]
#         outliers[cluster_centroid] = cluster_outliers
#     return outliers


root = Tk()
root.title("K-Means Clustering Analysis")

file_label = Label(root, text="Select File:")
file_label.grid(row=0, column=0, padx=5, pady=5)

file_entry = Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=5, pady=5)

file_button = Button(root, text="Browse", command=select_file)
file_button.grid(row=0, column=2, padx=5, pady=5)

percentage_label = Label(root, text="Percentage of Data to Read:")
percentage_label.grid(row=1, column=0, padx=5, pady=5)

percentage_entry = Entry(root, width=10)
percentage_entry.grid(row=1, column=1, padx=5, pady=5)
percentage_entry.insert(END, "100")

k_label = Label(root, text="Number of Clusters (K):")
k_label.grid(row=2, column=0, padx=5, pady=5)

k_entry = Entry(root, width=10)
k_entry.grid(row=2, column=1, padx=5, pady=5)
k_entry.insert(END, "3")

# std_multiplier_label = Label(root, text="Standard Deviation Multiplier:")
# std_multiplier_label.grid(row=3, column=0, padx=5, pady=5)
#
# std_multiplier_entry = Entry(root, width=10)
# std_multiplier_entry.grid(row=3, column=1, padx=5, pady=5)
# std_multiplier_entry.insert(END, "3")

start_button = Button(root, text="Start Analysis", command=start_analysis)
start_button.grid(row=4, column=1, padx=5, pady=5)

# Create a Frame to contain the Text and Scrollbar for clusters
clusters_frame = Frame(root)
clusters_frame.grid(row=5, column=0, padx=5, pady=5)

# Add a Text widget to display clusters
clusters_text = Text(clusters_frame, width=80, height=40, wrap=WORD)
clusters_text.pack(side=RIGHT, fill=Y)

# Add a Scrollbar to scroll through the clusters Text widget
clusters_scrollbar = Scrollbar(clusters_frame, orient=VERTICAL, command=clusters_text.yview)
clusters_scrollbar.pack(side=RIGHT, fill=Y)

# Link the Scrollbar to the clusters Text widget
clusters_text.config(yscrollcommand=clusters_scrollbar.set)

# Create a Frame to contain the Text and Scrollbar for outliers
outliers_frame = Frame(root)
outliers_frame.grid(row=5, column=1, padx=5, pady=5)

# Add a Text widget to display outliers
outliers_text = Text(outliers_frame, width=60, height=35, wrap=WORD)
outliers_text.pack(side=RIGHT, fill=Y)

# Add a Scrollbar to scroll through the outliers Text widget
outliers_scrollbar = Scrollbar(outliers_frame, orient=VERTICAL, command=outliers_text.yview)
outliers_scrollbar.pack(side=LEFT, fill=Y)

# Link the Scrollbar to the outliers Text widget
outliers_text.config(yscrollcommand=outliers_scrollbar.set)

# Create a Frame to contain the Text and Scrollbar for centroids

centroids_frame = Frame(root)
centroids_frame.grid(row=5, column=2, padx=2, pady=2)

# # Add a Label for centroids
# centroids_label = Label(centroids_frame, text="Centroids:")
# centroids_label.pack()

# Add a Text widget to display centroids
centroids_text = Text(centroids_frame, width=20, height=15, wrap=WORD)
centroids_text.pack(side=LEFT, fill=Y)

# Add a Scrollbar to scroll through the centroids Text widget
centroids_scrollbar = Scrollbar(centroids_frame, orient=VERTICAL, command=centroids_text.yview)
centroids_scrollbar.pack(side=RIGHT, fill=Y)

# Link the Scrollbar to the centroids Text widget
centroids_text.config(yscrollcommand=centroids_scrollbar.set)

root.mainloop()
