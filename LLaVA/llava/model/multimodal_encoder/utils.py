import math
import re
import time
import torch
import torch.nn as nn
import random
import numpy as np
import os
from types import SimpleNamespace

def normalize(tensor):
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)  # To avoid division by zero
    return tensor / norm


def normal_compute_pairwise_distances(X,l_2=False):
  
    X = torch.nn.functional.normalize(X,p=2.0,)
    
    dot_product = torch.mm(X, X.t())
 
    cosine_distance = 1 - dot_product
    
    return cosine_distance

class VIC:
    def __init__(self, dc=8, percentage=0.08, k=0.6):
        """
        Vision Information Clustering
        Vision Information Clustering algorithm.
        
        Args:
            dc (float): Cutoff distance threshold for local density calculation.
            percentage (float): Fixed percentage of points to be selected as cluster centers.
        """
        self.dc = dc  # Cutoff distance threshold for local density calculation
        self.percentage = percentage  # Percentage for selecting cluster centers
        self.k = k  # Spatial distance threshold

    def get_clusters(self):
        """
        Get the clustering results.
        
        Returns:
            dict: A dictionary mapping each cluster label to its sample indices.
        """
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("The clustering algorithm must be run first.")
        
        # Return each cluster and the sample indices it contains
        clusters = {label: np.where(self.labels_ == label)[0] 
                    for label in np.unique(self.labels_)}
        return clusters

    def fit(self, X, pos=None):
        """
        Perform Vision Information Clustering.
        
        Args:
            X (torch.Tensor): Feature input [n_samples, n_features].
            pos (torch.Tensor): Coordinate input [n_samples, 2], normalized (x, y) coordinates.
        """
        n_samples = X.shape[0]    
        # Handle empty input case
        if n_samples == 0:
            self.labels_ = np.array([], dtype=int)
            return
        device = X.device

        # === Compute distance matrix ===
        # Calculate Euclidean distance in the feature space
        dist_feat = self._compute_pairwise_distances(X)
        dist_feat = torch.clamp(dist_feat, min=0.0)
        dist = dist_feat
        dist.fill_diagonal_(0)  # Set self-distance to 0

        # === Core DPC algorithm ===
        # 1. Calculate local density rho
        # Calculate local density for each point using a Gaussian kernel
        rho = torch.exp(-(dist / self.dc) ** 2).sum(dim=1)

        # === 2. Calculate delta (minimum distance to a higher density point, with spatial constraints) ===
        # Construct density comparison matrix: rho_compare[i,j] = True means density of j is not higher than i
        rho_expand = rho.unsqueeze(1).expand(n_samples, n_samples)
        rho_compare = ~torch.gt(rho_expand, rho_expand.t())
        rho_compare.fill_diagonal_(False)  # Exclude self-comparison

        # Initialize spatial distance matrix
        if pos is not None:
            dist_spatial = self._compute_pairwise_distances(pos)
            dist_spatial = torch.clamp(dist_spatial, min=0.0)
            # Add spatial distance constraint: only consider points with spatial distance <= k
            spatial_mask = dist_spatial <= self.k
            # Combine density and spatial constraints: higher density and spatially close
            valid_mask = rho_compare & spatial_mask
            # Mask distances that do not meet the criteria with inf
            inf_mask = torch.full_like(dist, float('inf'))
            conditioned_dist = torch.where(valid_mask, dist, inf_mask)
            # Calculate delta and the index of the nearest higher-density point
            delta, nearest_higher_density_indices = torch.min(conditioned_dist, dim=1)
        else:
            # No spatial constraint, use original feature distance + density
            inf_mask = torch.full_like(dist, float('inf'))
            conditioned_dist = torch.where(rho_compare, dist, inf_mask)
            delta, nearest_higher_density_indices = torch.min(conditioned_dist, dim=1)

        # 3. Calculate importance metric gamma = delta × rho
        # Points with larger gamma values are more likely to be cluster centers
        gamma = delta * rho
        gamma_sorted_indices = torch.argsort(-gamma)  # Sort by gamma value in descending order

        # 4. Select cluster centers
        # Select points with the largest gamma values as cluster centers based on the set percentage
        num_cluster_centers = max(1, int(self.percentage * n_samples))
        cluster_centers = gamma_sorted_indices[:num_cluster_centers]

        # 5. Label propagation
        # Initialize labels to -1
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=device)
        
        # Cluster center indices, set their labels (i.e., the index of the center point)
        cluster_centers = cluster_centers.to(device)
        labels[cluster_centers] = torch.arange(len(cluster_centers), device=device)

        # Iteratively propagate labels until no new updates occur (parallel approach)
        # Create a mask to mark whether propagation should continue
        unassigned_mask = (labels == -1)
        updated = True

        while updated:
            # Find the corresponding higher-density neighbors for points that have not yet been assigned a label
            to_update = unassigned_mask & (labels[nearest_higher_density_indices] != -1)

            if not to_update.any():
                updated = False
                break

            # Assign the label of the higher-density neighbor to the current unlabeled point
            src_labels = labels[nearest_higher_density_indices[to_update]]
            labels[to_update] = src_labels

            # Update the mask
            unassigned_mask = (labels == -1)

        # Re-ensure that the labels of the cluster centers have not been changed
        labels[cluster_centers] = torch.arange(len(cluster_centers), device=device)

        # Convert the result to a numpy array and save it
        self.labels_ = labels.cpu().numpy()

    def _compute_pairwise_distances(self, X):
        """
        Compute pairwise Euclidean distances in the feature space.
        
        Args:
            X (torch.Tensor): Input features [n_samples, n_features].
            
        Returns:
            torch.Tensor: Distance matrix [n_samples, n_samples].
        """
        # Use broadcasting to calculate the Euclidean distance between all pairs of points
        X_expanded = X.unsqueeze(1)  # [n_samples, 1, n_features]
        X_transposed = X.unsqueeze(0)  # [1, n_samples, n_features]
        
        # Calculate the squared Euclidean distance
        dist_squared = torch.sum((X_expanded - X_transposed) ** 2, dim=2)
        
        # Return the Euclidean distance
        return torch.sqrt(dist_squared + 1e-8)  # Add a small constant to avoid numerical instability

    def get_cluster_info(self):
        """
        Get detailed information about the clustering.
        
        Returns:
            dict: A dictionary containing information such as the number of clusters, number of noise points, etc.
        """
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("The clustering algorithm must be run first.")
        
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels)
        n_noise = np.sum(self.labels_ == -1) if -1 in unique_labels else 0
        
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[label] = np.sum(self.labels_ == label)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'total_samples': len(self.labels_)
        }