# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     20 June 2023
#
# Purpose:  Network model in pytorch for DDPG within webots
#
# References
# ----------
#
# This implementation based on kmeans functions from:
# https://github.com/Sweety-dm/Interference-aware-Deep-Q-learning/blob/master/IQ-RE/IQ_dqn_agent.py
#
import typing

import numpy as np
import torch


def calculate_centroid(centroids: list, vector: list) -> int:
    """Calculate which centroid the vector belongs to based on euclidian distance.
    Return the closest centroid."""
    distances = [np.linalg.norm(np.subtract(vector, centroid)) for centroid in centroids]
    return int(np.argmin(distances))


def update_centroids(centroids: list, centroid_counts: list, vector: list, centroid: int) -> list:
    """Method to get the adjusted centroids and cluster based on the vector given."""
    centroids[centroid] += (1 / centroid_counts[centroid]) * (np.subtract(vector, centroids[centroid]))

    return centroids



