class cluster:
    """Represents a cluster with number, coordinates, and points.

    Attributes:
        numero (int): The number of the cluster.
        points (list): List of points in the cluster.
        coord (list): List of coordinates of the cluster.
        moyenne (list): List to store the mean values of coordinates.

    Methods:
        __init__(self, number, coord_cluster): Initializes the cluster with the given number and coordinates.
        add(self, point): Adds a point to the cluster.
        means(self): Calculates the mean values of the points in the cluster.
        clear(self): Clears all points from the cluster.
    """

    def __init__(self, number, coord_cluster):
        """Initializes the cluster with the provided number and coordinates.

        Args:
            number (int): The number of the cluster.
            coord_cluster (list): List of coordinates of the cluster.
        """
        self.numero: int  = number
        self.points: list = []
        self.coord: list = coord_cluster
        self.moyenne: list = []

    def add(self, point):
        """Adds a point to the cluster.

        Args:
            point (list): The point to be added to the cluster.
        """
        self.points.append(point)

    def means(self):
        """Calculates the mean values of the points in the cluster."""
        nouv_coord = []
        nombre = len(self.coord)
        if len(self.points) == 0:
            for i in range(nombre):
                nouv_coord.append(0)
            self.coord = nouv_coord
        else:
            self.moyenne = self.coord
            self.coord = []
            for i in range(nombre):
                a = 0
                for u in self.points:
                    a += u[i]
                a = a / len(self.points)
                nouv_coord.append(a)
            self.coord = nouv_coord

    def clear(self):
        """Clears all points from the cluster."""
        self.points = []

import math
import random
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np

def distance_euclidean(point1, point2):
    """Calculates the Euclidean distance between two points in n-dimensional space.

    Args:
        point1 (list): The coordinates of the first point.
        point2 (list): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    If the dimensions of the two points are not the same, a message is printed indicating the mismatch.
    The Euclidean distance is calculated as the square root of the sum of the squared differences between coordinates of the two points.
    The Euclidean distance is a measure of the straight-line distance between two points in n-dimensional space.
    """
    # Check if the two points have the same number of dimensions
    if len(point1) != len(point2):
        # Print a message indicating the mismatch
        print("The two points do not have the same dimensions. #point1=", len(point1), "#point2=", len(point2))
    else:
        # Initialize a variable to store the sum of the squared differences between coordinates of the two points
        distance = 0
        # Calculate the sum of the squared differences between coordinates of the two points
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        # Return the Euclidean distance, which is the square root of the sum of the squared differences between coordinates of the two points
        return math.sqrt(distance)


def distance_manhattan(point1, point2):
    """Calculates the Manhattan distance between two points represented by their coordinates.

    The Manhattan distance is the sum of the absolute differences between the coordinates of the two points.
    It is commonly used in graph theory and machine learning.

    Args:
        point1 (list): The coordinates of the first point.
        point2 (list): The coordinates of the second point.

    Returns:
        int: The Manhattan distance between the two points.

    This function calculates the Manhattan distance between two points in n-dimensional space. 
    The Manhattan distance is defined as the sum of the absolute differences.
    It is a measure of the distance between two points in the space of coordinates.
    If they do, it calculates the Manhattan distance by summing the absolute differences in coordinates.
    If they do not, it prints a message indicating the mismatch.
    """
    # Check if the two points have the same number of dimensions
    if len(point1) != len(point2):
        # Print a message indicating the mismatch
        print("The two points do not have the same dimensions. #point1=", len(point1), "#point2=", len(point2))
    elif len(point1) == len(point2):
        # Initialize a variable to store the distance
        distance = 0
        # Calculate the distance by summing the absolute differences in coordinates
        for i in range(len(point1)):
            distance += abs(point1[i] - point2[i])
        # Return the distance
        return distance
    
p = 3
def distance_lp(point1, point2, p):
    """
    Calculates the L^p distance between two points represented by their coordinates.

    Args:
        point1 (list): The coordinates of the first point.
        point2 (list): The coordinates of the second point.
        p (int): The value of p for L^p distance calculation.

    Returns:
        float: The L^p distance between the two points.

    This function calculates the L^p distance between two points in n-dimensional space. 
    The L^p distance is defined as the p-th root of the sum of the absolute differences 
    raised to the power of p. It is a measure of the distance between two points in 
    the space of coordinates.

    If the dimensions of the two points are not the same, a message is printed 
    indicating the mismatch.
    """
    # Check if the two points have the same number of dimensions
    if len(point1) != len(point2):
        # Print a message indicating the mismatch
        print("The two points do not have the same dimensions. #point1=", len(point1), "#point2=", len(point2))
    elif len(point1) == len(point2):
        # Initialize a variable to store the sum of the absolute differences raised to the power of p
        k = 0
        # Calculate the sum of the absolute differences raised to the power of p
        for i in range(len(point1)):
            k += abs(point1[i] - point2[i]) ** p
        # Return the L^p distance, which is the p-th root of the sum of the absolute differences raised to the power of p
        return k ** (1/p)
    
def distance_cos(point1, point2):
    """
    Calculates the cosine distance between two points.

    Args:
        point1 (list): The coordinates of the first point.
        point2 (list): The coordinates of the second point.

    Returns:
        float: The cosine distance between the two points.

    This function calculates the cosine distance between two points in n-dimensional space. 
    The cosine distance is defined as the cosine of the angle between the two points. 
    It is a measure of the distance between two points in the space of coordinates.
    If the dimensions of the two points are not the same, a message is printed indicating the mismatch.
    """
    if len(point1) != len(point2):
        print("The two points do not have the same dimensions. #point1=", len(point1), "#point2=", len(point2))
    elif len(point1) == len(point2):
        # Initialize variables
        k = 0  # sum of the products of corresponding coordinates
        x_sum = 0  # sum of the squares of coordinates of the first point
        y_sum = 0  # sum of the squares of coordinates of the second point

        # Calculate the sum of the products of corresponding coordinates
        for i in range(len(point1)):
            k += point1[i] * point2[i]

        # Calculate the sum of the squares of coordinates of the first point
        for i in range(len(point1)):
            x_sum += point1[i] ** 2

        # Calculate the sum of the squares of coordinates of the second point
        for i in range(len(point2)):
            y_sum += point2[i] ** 2

        # Check if the points are on the same axis (i.e., one or both of the points have all coordinates equal to 0)
        if math.sqrt(x_sum) * math.sqrt(y_sum) == 0:
            return 0

        # Calculate the cosine distance
        return k / (math.sqrt(x_sum) * math.sqrt(y_sum))

def distance_jaccard(point1, point2):
    ''' 
    Calculate the Jaccard distance between two points represented as binary vectors.

    Args:
        point1 (list): Binary vector representing the first point.
        point2 (list): Binary vector representing the second point.
    
    Returns:
        float: Jaccard distance between the two points.

    This function calculates the Jaccard distance between two points in n-dimensional space.
    The Jaccard distance is a measure of similarity between two sets, defined as the size
    of the intersection divided by the size of the union of the sets.
    '''
    
    # Initialize variables to keep track of the intersection and the sizes of the sets
    intersection = 0
    A = 0  # Size of set A
    B = 0  # Size of set B
    
    # Check if the two points have the same number of dimensions
    if len(point1) != len(point2):
        # Print a message indicating the mismatch
        print("The two points do not have the same dimensions. #point1=", len(point1), "#point2=", len(point2))
    elif len(point1) == len(point2):
        # Iterate over the coordinates of both points
        for i in range(len(point1)):
            # If both coordinates are 1, increase the intersection count
            if point1[i] == 1 and point2[i] == 1:
                intersection += 1
            # If the first coordinate is 1, increase the size of set A
            elif point1[i] == 1:
                A += 1
            # If the second coordinate is 1, increase the size of set B
            elif point2[i] == 1:
                B += 1
        
        # Check if the intersection is 0, indicating that the sets are disjoint
        if A + B - intersection == 0:
            return 0
        
    # Calculate the Jaccard distance by dividing the intersection by the size of the union
    return intersection / (A + B - intersection)


def choose_centers(k, points):
    '''
    Chooses `k` random points from the `points` list and returns them as a list of lists.
    Each inner list contains the coordinates of a center point.

    Args:
        k (int): The number of center points to choose.
        points (list): The list of points from which to choose center points.

    Returns:
        list: A list of lists, where each inner list contains the coordinates of a center point.

    This function chooses `k` random points from the `points` list and returns them as a list of lists.
    Each inner list contains the coordinates of a center point. The points are chosen randomly.
    '''
    # Initialize an empty list to store the chosen center points
    centers = []
    # Initialize a variable to keep track of the number of centers that have been selected
    num_centers_selected = 0

    # Keep selecting random points from the `points` list until `k` centers have been selected
    while num_centers_selected < k:
        # Generate a random index within the range of valid indices for the `points` list
        random_index = random.randint(0, len(points) - 1)
        # Check if the point at the randomly generated index has not already been selected as a center
        if points[random_index] not in centers:
            # If not, add the point to the list of chosen center points
            center = points[random_index]
            centers.append(center)
            # Increment the counter of selected centers
            num_centers_selected += 1

    # Return the list of chosen center points
    return centers

def choose_centers_plus(k, points, distance):
    '''
    Chooses `k` centers from the given `points` list using a probabilistic approach.
    
    Args:
        k (int): The number of centers to choose.
        points (list): A list of lists representing the coordinates of the points.
        distance (function): A function that calculates the distance between two points.
    
    Returns:
        list: A list of lists representing the coordinates of the chosen centers.

    This function chooses `k` centers from the given `points` list using a probabilistic approach.
    The centers are selected using the following steps:
    1. Select a random point from the `points` list.    
    2. If the point is not already a center, add it to the list of centers.
    3. If the number of centers has reached `k`, return the list of centers.
    4. Repeat steps 1-3 until all `k` centers have been selected.
    '''
    centers = {}
    random_index = random.randint(0, len(points) - 1)
    centers[0] = cluster(0, points[random_index])

    while len(centers) != k:
        distances = [[point, min([distance(point, centers[cluster].coord) for cluster in centers])] for point in points]

        total_distance = sum(distances[i][1] for i in range(len(distances)))
        probabilities = [distances[i][1] / total_distance for i in range(len(distances))]

        random_index = random.choices(range(len(points)), weights=probabilities, k=1)[0]

        center_id = len(centers)
        centers[center_id] = cluster(center_id, distances[random_index][0])

    return [centers[cluster].coord for cluster in centers]


# liste_des_points=[[15, 20],[20, 16],[10, 14],[15, 14],[19, 19],[7, 0],[18, 13],[12, 5],[4, 19],[9, 14],[16, 9],[10, 12],[19, 18],[8, 17],[10, 20],[11, 4],[14, 1],[11, 3],[8, 18],[11, 13],[4, 12],[7, 16],[1, 6],[12, 7],[12, 0],[9, 3],[14, 5],[18, 18],[10, 8],[4, 12]]
# k=3
# 
# dico_centre_plus = choose_centers_plus(k,liste_des_points,  distance_euclidean)    
# dico_centre = choose_centers(k,liste_des_points)
# print(dico_centre_plus)
# print(dico_centre)
# 
# cluster_plus_x=[]
# cluster_plus_y=[]
# cluster_x=[]
# cluster_y=[]
# 
# for i in range(k):
#     cluster_plus_x.append(dico_centre_plus[i][0])
#     cluster_plus_y.append(dico_centre_plus[i][1])
#     cluster_x.append(dico_centre[i][0])
#     cluster_y.append(dico_centre[i][1])
#     
# plt.scatter(cluster_plus_x,cluster_plus_y,c='red',label='méthode Kmeans++')
# plt.scatter(cluster_x,cluster_y,c='blue',label='méthode Kmeans')
# plt.ylabel('Coordonnées en Y')
# plt.xlabel('Coordonnées en X')
# plt.xlim(-5,25)
# plt.ylim(-5,25)
# plt.legend()
# plt.show()

def k_means(k, points, kmeansplus, distance, iterations):
    """
    Takes in k (the number of centroids), the list of points, and the number
    of iterations that should occur.
    Returns a list containing the points of each cluster.

    Args:
        k (int): Number of centroids.
        points (list): List of points.
        kmeansplus (bool): Whether to use the K-means++ algorithm.
        distance (function): Function to calculate the distance between two points.
        iterations (int): Number of iterations.

    Returns:
        dict: Dictionary containing the points of each cluster, where the keys are
              the centroids and the values are the lists of points.
    """

    # Choose the initial centers using the K-means++ algorithm if requested.
    if kmeansplus is False:
        centers = choose_centers(k, points)
    else:
        centers = choose_centers_plus(k, points, distance)

    #print(centers)
    # Create a dictionary to store the clusters.
    clusters = {i: cluster(i, center) for i, center in enumerate(centers)}

    # Perform the k-means clustering.
    for _ in range(iterations):
        # Clear the points of each cluster.
        for cluster1 in clusters.values():
            cluster1.clear()

        # Assign each point to the nearest cluster.
        for point in points:
            min_distance = float('inf')
            min_center = None

            for i,center in enumerate(centers):
                distance1 = distance(point, center)
                if distance1 < min_distance:
                    min_distance = distance1
                    min_center = i

            # print("ONEEB the value of cluster are", clusters)
            # print("ONEEB the value of min_center is", min_center)

            clusters[min_center].add(point)

        # Calculate the centroid of each cluster.
        for cluster1 in clusters.values():
            cluster1.means()

    return clusters


def calculate_intra_inertia(points, clusters):
    """
    Calculates the intra-cluster inertia.

    Intra-cluster inertia is a measure of how spread out the points in each cluster are.
    It is calculated as the sum of the squared Euclidean distances between each point and the centroid of its cluster.
    The intra-cluster inertia is then divided by the total number of points to obtain the average intra-cluster inertia.

    Args:
        points (list): List of points.
        clusters (dict): Dictionary containing the points of each cluster, where the keys are
                         the centroids and the values are the lists of points.

    Returns:
        float: The average intra-cluster inertia.
    """
    total_inertia = 0.0  # Initialize the total intra-cluster inertia to 0.0

    # Iterate over each cluster
    for cluster in clusters.values():
        # Iterate over each point in the cluster
        for point in cluster.points:
            # Calculate the Euclidean distance between the point and the centroid of its cluster
            distance_to_centroid = distance_euclidean(point, cluster.coord)
            # Add the squared distance to the total intra-cluster inertia
            total_inertia += distance_to_centroid ** 2

    # Calculate the average intra-cluster inertia by dividing the total intra-cluster inertia by the total number of points
    average_intra_inertia = total_inertia / len(points)

    return average_intra_inertia

def calculate_inter_inertia(points, clusters):
    """
    Calculates the inter-cluster inertia.

    Inter-cluster inertia is a measure of how spread out the clusters are.
    It is calculated as the sum of the squared Euclidean distances between each centroid
    and the center of mass (center of gravity) of the points in the dataset.
    The inter-cluster inertia is then divided by the total number of points to obtain the average inter-cluster inertia.

    Args:
        points (list): List of points.
        clusters (dict): Dictionary containing the points of each cluster, where the keys are
                         the centroids and the values are the lists of points.

    Returns:
        float: The average inter-cluster inertia.
    """
    # Calculate the center of mass for each dimension
    center_of_mass = []  # List to store the center of mass for each dimension
    num_points = len(points)  # Number of points in the dataset
    num_dimensions = len(points[0])  # Number of dimensions in the dataset
    
    # Calculate the center of mass for each dimension
    for i in range(num_dimensions):
        total = 0  # Initialize the total sum for the current dimension
        for point in points:
            total += point[i]  # Add the current value to the total sum
        center_of_mass.append(total / num_points)  # Append the center of mass to the list
    
    inter_inertia = 0  # Initialize the inter-cluster inertia to 0
    
    # Calculate the inter-cluster inertia
    for i, centroid in enumerate(clusters):
        num_points_in_cluster = len(clusters[centroid].points)  # Number of points in the current cluster
        distance_sum = 0  # Initialize the sum of distances to 0
        for point in clusters[centroid].points:
            distance_sum += distance_euclidean(point, center_of_mass)  # Calculate and add the distance to the sum
            
        inter_inertia += num_points_in_cluster * distance_sum  # Calculate the inter-cluster inertia for the current cluster
    
    return inter_inertia / num_points  # Return the average inter-cluster inertia

def coude(data, nb_de_cluster, distance, kmeansplus, nb_diteration):
    """
    Plot the curve of total inertia versus number of clusters.

    This function runs k-means with different number of clusters and calculates the total
    inertia (sum of intra-cluster inertia and inter-cluster inertia) for each number of clusters.
    It then plots the curve of total inertia versus number of clusters.

    Args:
        data (list): List of points.
        nb_de_cluster (int): Maximum number of clusters.
        distance (function): Function to calculate the distance between two points.
        kmeansplus (bool): Whether to use the K-means++ algorithm.
        nb_diteration (int): Number of iterations.
    """
    # Initialize lists to store the results
    total_inertia_list = []  # List to store the total inertia
    cluster_number_list = []  # List to store the number of clusters

    # Iterate over the range of clusters
    for i in range(1, nb_de_cluster):
        # Run k-means with the current number of clusters
        clusters = k_means(i, data, kmeansplus, distance, nb_diteration)
        # Calculate the total inertia and append it to the list
        total_inertia = calculate_intra_inertia(data, clusters) + calculate_inter_inertia(data, clusters)
        total_inertia_list.append(total_inertia)
        # Append the current number of clusters to the list
        cluster_number_list.append(i)

    # Plot the curve
    plt.scatter(cluster_number_list, total_inertia_list)
    plt.plot(cluster_number_list, total_inertia_list)
    plt.ylabel('Total inertia')
    plt.xlabel('Number of clusters K')
    plt.show()


 
