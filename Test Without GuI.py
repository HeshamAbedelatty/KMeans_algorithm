from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from random import sample

# Calculate the total number of rows in the file
total_rows = sum(1 for _ in open("imdb_top_2000_movies.csv"))

# Calculate the number of rows to read (70% of total rows)
nrows_to_read = int(total_rows * 1.0)

# Read only the 'TransactionNo' and 'Items' columns from the CSV file
data = pd.read_csv("imdb_top_2000_movies.csv", usecols=['IMDB Rating'], nrows=nrows_to_read)

def find_outliers(movies_cleaned):
    x1 = movies_cleaned['IMDB Rating'].quantile(0.25)
    x3 = movies_cleaned['IMDB Rating'].quantile(0.75)
    print(x1)
    print(x3)
    IQR = x3 - x1
    l = x1 - 1.5 * IQR  # lower
    u = x3 + 1.5 * IQR  # upper
    print(l)
    print(u)
    outliers_iqr = movies_cleaned[(movies_cleaned['IMDB Rating'] < l) | (movies_cleaned['IMDB Rating'] > u)]

    movies_cleaned = movies_cleaned[(movies_cleaned['IMDB Rating'] >= l) & (movies_cleaned['IMDB Rating'] <= u)]

    return movies_cleaned, outliers_iqr

MovieData,outliers = find_outliers(data)
RatesData = []
for index, row in MovieData.iterrows():
    # Extract the transaction ID and item
    rate = row['IMDB Rating']
    RatesData.append(rate)


# Initialize centroids randomly
def initialize_centroids(c, data):
    while True:
        Centroids = sample(list(data), c)
        if len(Centroids) == len(set(Centroids)):
            break
    return Centroids


k = 3  # Number of clusters
centroids = initialize_centroids(k, RatesData)


# print(centroids)


# Function to calculate Euclidean distance
def euclidean_distance(point, centriod):
    return np.sqrt(np.sum((point - centriod) ** 2))


# Function to assign data points to nearest centroid
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


result = assign_to_clusters(RatesData, centroids)


# print(result)

# Function to update centroids
def update_centroids(clusters):
    # clusters is {centriod: list of points like:[7.0, 6.8, 6.7]}
    centroids = []
    for centroid, cluster_points in clusters.items():
        #  # Calculate the mean of cluster points (float points)
        mean_point = np.mean(cluster_points)
        # print(mean_point)
        # mean_point = round(mean_point, 1)
        centroids.append(mean_point)
    return centroids


# calculate the new centroid by mean
new_centriod = update_centroids(result)


# print(new_centriod)


def kmeans_clustering(data, k):
    # Initialize centroids randomly
    centroids = initialize_centroids(k, RatesData)

    while True:
        # Assign data points to clusters
        clusters = assign_to_clusters(data, centroids)

        # Update centroids
        new_centroids = update_centroids(clusters)

        # Check for convergence
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return centroids, clusters


centroids, clusters = kmeans_clustering(RatesData, k)

print("Centroids: ")
print(centroids)

# print("\nClusters: ")
# for key, values in clusters.items():
#     print()
#     print("Key:", key)
#     for value in values:
#         ifMultiple = MovieData[MovieData['IMDB Rating'] == value]
#
#         if len(ifMultiple) > 1:
#             for i in list(ifMultiple):
#                 pd.set_option('display.max_columns', None)
#                 print(i)
#         pd.set_option('display.max_columns', None)
#         print(ifMultiple)


# print(clusters)


# Step 8: Detect outliers

# Define a function to detect outliers
# def detect_outliers(clusters):
#     # Initialize an empty list to store outliers
#     Outliers = {}
#     # Loop through each cluster
#     for cluster_centriod, cluster_points in clusters.items():
#         # Calculate the mean and standard deviation of the cluster
#         mean = np.mean(cluster_points)
#         std = np.std(cluster_points)
#
#         # Set a threshold for outliers (3 standard deviations from the mean)
#         outlier_threshold = mean + 4 * std
#
#         # Find data points in the cluster that exceed the outlier threshold
#         cluster_outliers = [point for point in cluster_points if point > outlier_threshold]
#
#         # Add detected outliers to the outliers list
#         Outliers[cluster_centriod] = cluster_outliers
#
#     # Return the list of outliers
#     return Outliers
def detect_outliers(data):
    # Initialize an empty list to store outliers

    # Loop through each cluster
    mean_rating = data['IMDB Rating'].mean()
    std_dev_rating = data['IMDB Rating'].std()
    outlier_threshold = mean_rating - 3 * std_dev_rating
    print(outlier_threshold)
    # Find data points in the cluster that exceed the outlier threshold
    # cluster_outliers = data[(data['IMDB Rating'] < outlier_threshold) | (data['IMDB Rating'] > outlier_threshold)]
    cluster_outliers = data[(data['IMDB Rating'] < outlier_threshold)]
    #cluster_outliers = [point for point in cluster_points if point > outlier_threshold]

    # Return the list of outliers
    return cluster_outliers


# Call the detect_outliers function with the standardized data and cluster assignments
requierdmovies,outliers = find_outliers(data)
# outliers = detect_outliers(data)
print("Outliers:")
print(outliers)
# print(requierdmovies)
# for key, value in outliers.items():
#     print("Key:", key)
#     print("Value:", value)
